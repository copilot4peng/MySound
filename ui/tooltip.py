"""Small non-modal labels for compact icon toolbars."""
import tkinter as tk


def attach_tooltip(widget, app, source):
    timer = None
    window = None

    def hide(_event=None):
        nonlocal timer, window
        if timer is not None:
            widget.after_cancel(timer)
            timer = None
        if window is not None:
            window.destroy()
            window = None

    def show():
        nonlocal timer, window
        timer = None
        if not widget.winfo_exists():
            return
        window = tk.Toplevel(widget)
        window.overrideredirect(True)
        p = app.theme.palette
        tk.Label(window, text=source() if callable(source) else app.tr(source), bg=p["surface"], fg=p["text"],
                 wraplength=min(640, widget.winfo_screenwidth() - 40), justify="left",
                 font=app.font("small"), padx=10, pady=6, relief="solid", borderwidth=1).pack()
        window.update_idletasks()
        x = min(widget.winfo_rootx(), max(0, widget.winfo_screenwidth() - window.winfo_reqwidth()))
        y = widget.winfo_rooty() + widget.winfo_height() + 5
        if y + window.winfo_reqheight() > widget.winfo_screenheight():
            y = widget.winfo_rooty() - window.winfo_reqheight() - 5
        window.geometry(f"+{max(0, x)}+{max(0, y)}")

    def schedule(_event=None):
        nonlocal timer
        hide()
        timer = widget.after(450, show)

    widget.bind("<Enter>", schedule, add="+")
    widget.bind("<Leave>", hide, add="+")
    widget.bind("<ButtonPress>", hide, add="+")
    widget.bind("<Destroy>", hide, add="+")
