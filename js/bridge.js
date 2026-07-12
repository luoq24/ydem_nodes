/**
 * ydem_nodes - Prompt Manager Bridge (v8)
 *
 * 在 ComfyUI 页面内轮询本地 prompt-manager 服务，把收到的文本填入指定节点的 widget。
 * 只填【激活工作流】的节点；若指定了 workflowKey 且它未激活，则提示用户去点亮该 tab。
 * （不自动切换工作流——openWorkflow 实测会破坏目标工作流内容，已禁用）
 *
 * 协议：
 *   GET  /task  -> { pending, id, text, nodeTitle, widgetName }
 *   POST /ack   <- { id, ok, error }
 *
 * 改 JS 后只需浏览器硬刷新（Ctrl+Shift+R），无需重启 ComfyUI。
 */
import { app } from "/scripts/app.js";

const SERVER = "http://127.0.0.1:9223";
const POLL_MS = 700;        // 轮询间隔
const RETRY_MS = 600;       // 暂未就绪时的重试间隔
const MAX_RETRY = 6;        // 重试次数

let processingId = null;

app.registerExtension({
    name: "ydem.PromptManagerBridge",
    setup() {
        console.log("[ydem-bridge] v8 已加载，开始轮询", SERVER);
        setInterval(poll, POLL_MS);
        poll();
    },
});

/** 轮询本地服务，拿到待处理任务就执行 */
async function poll() {
    try {
        const res = await fetch(`${SERVER}/task`, { cache: "no-store" });
        if (!res.ok) return;
        const task = await res.json();
        if (!task || !task.pending || !task.id) return;
        if (task.id === processingId) return; // 正在处理中
        processingId = task.id;
        runTaskWithRetry(task, 0);
    } catch (e) {
        // 服务端没开 / 网络错误，静默
    }
}

async function runTaskWithRetry(task, attempt) {
    let result;
    try {
        result = await handleTask(task);
    } catch (e) {
        result = { ok: false, transient: false, error: "handleTask异常: " + String(e?.message || e) };
    }
    if (result.ok) {
        ack(task.id, true, null);
        processingId = null;
        return;
    }
    if (attempt < MAX_RETRY && result.transient) {
        setTimeout(() => runTaskWithRetry(task, attempt + 1), RETRY_MS);
        return;
    }
    console.warn("[ydem-bridge] 失败:", result.error);
    ack(task.id, false, result.error);
    processingId = null;
}

/** 找节点并填入。只填【激活工作流】的节点；指定 workflowKey 时要求它必须已激活。 */
async function handleTask(task) {
    const nodeTitle = task.nodeTitle || "手动提示词";
    const widgetName = task.widgetName || "text";
    const workflowKey = task.workflowKey || "";
    const text = task.text ?? "";

    const wfMgr = app.extensionManager?.workflow;

    // ComfyUI 尚未就绪 -> 重试
    if (!app.graph?._nodes && !(wfMgr?.openWorkflows?.length)) {
        return { ok: false, transient: true, error: "ComfyUI 图尚未就绪" };
    }

    // 指定了目标工作流：必须它是当前激活的才填（自动切换会破坏数据，已禁用）
    if (workflowKey) {
        const target = findWorkflow(wfMgr, workflowKey);
        if (!target) {
            return { ok: false, transient: false, error: `工作流未打开: ${workflowKey}（请在 ComfyUI 里打开它）` };
        }
        if (wfMgr.activeWorkflow?.key !== target.key) {
            return { ok: false, transient: false, error: `目标工作流「${workflowKey}」未激活，请先在 ComfyUI 点亮它的 tab，再点填入` };
        }
    }

    // 在激活工作流里找节点并填入
    const node = await waitForActiveNode(nodeTitle, 1200);
    if (node) return setWidgetOnNode(node, widgetName, text, nodeTitle);
    const titles = (app.graph?._nodes || []).map(n => n.title || n.type).slice(0, 20).join(", ") || "(画布无节点)";
    return { ok: false, transient: false, error: `找不到节点: ${nodeTitle}。当前画布: ${titles}` };
}

/** 在激活工作流里轮询找节点，超时返回 null */
function waitForActiveNode(title, timeout) {
    return new Promise((resolve) => {
        const start = Date.now();
        const tick = () => {
            const node = (app.graph?._nodes || []).find(n => n.title === title);
            if (node) return resolve(node);
            if (Date.now() - start >= timeout) return resolve(null);
            setTimeout(tick, 120);
        };
        tick();
    });
}

/** 在已打开工作流里按名字找（兼容带/不带 .json、filename 等多种写法） */
function findWorkflow(wfMgr, name) {
    const list = wfMgr?.openWorkflows || [];
    return list.find(w =>
        w.key === name
        || w.key === name + ".json"
        || w.filename === name
        || w.fullFilename === name
    ) || null;
}

/** 在节点上找到 widget 并设值 */
function setWidgetOnNode(node, widgetName, text, nodeTitle) {
    const widget = (node.widgets || []).find(w => w.name === widgetName);
    if (!widget) {
        const names = (node.widgets || []).map(w => w.name).join(", ") || "(无 widget)";
        return { ok: false, transient: false, error: `节点 ${nodeTitle} 上找不到 widget: ${widgetName}。现有: ${names}` };
    }
    try {
        setWidgetValue(node, widget, text);
        console.log(`[ydem-bridge] 已填入 ${nodeTitle}.${widgetName}（${text.length} 字）`);
        return { ok: true };
    } catch (e) {
        return { ok: false, transient: false, error: "setWidgetValue异常: " + String(e?.message || e) };
    }
}

/** 设置 widget 值：设 value + 同步 DOM + 触发回调 + 标脏，多管齐下 */
function setWidgetValue(node, widget, text) {
    widget.value = text;

    // DOM widget：同步可见的 textarea/input 并派发事件
    const el = widget.inputEl
        || (widget.element && widget.element.querySelector && widget.element.querySelector("textarea, input"));
    if (el) {
        el.value = text;
        el.dispatchEvent(new Event("input", { bubbles: true }));
        el.dispatchEvent(new Event("change", { bubbles: true }));
    }

    if (typeof widget.callback === "function") {
        try { widget.callback.call(widget, text, node); } catch (e) {}
    }
    if (typeof node.setDirtyCanvas === "function") node.setDirtyCanvas(true, true);
    if (typeof node.onWidgetChanged === "function") {
        try { node.onWidgetChanged(widget.name, text, text); } catch (e) {}
    }
}

async function ack(id, ok, error) {
    try {
        await fetch(`${SERVER}/ack`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id, ok, error }),
        });
    } catch (e) {}
}
