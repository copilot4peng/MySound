"""Keep desktop error dialogs readable; the logger retains full diagnostics."""


def concise_error(message, translate=lambda source: source):
    message = str(message).strip()
    lower = message.lower()
    if ("cuda" in lower and "out of memory" in lower) or "cuda 显存不足" in lower:
        return translate("CUDA 显存不足。请选择自动模式（显存不足时改用 CPU）、直接使用 CPU，或选择更小的模型。")
    if len(message) > 450:
        return message[:350].rstrip() + "…\n" + translate("当前错误信息过长，请查看日志中的完整内容。")
    return message
