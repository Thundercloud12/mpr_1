import tkinter as tk
from tkinter import ttk


class RunLogSection:
    def __init__(self, parent: ttk.Frame) -> None:
        log_frame = ttk.LabelFrame(parent, text="Run Log", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_output = tk.Text(log_frame, height=18, font=("Courier New", 11), wrap="word")
        self.log_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(log_frame, command=self.log_output.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_output.config(yscrollcommand=scrollbar.set)

    def append(self, message: str) -> None:
        self.log_output.insert(tk.END, message + "\n")
        self.log_output.see(tk.END)

    def clear(self) -> None:
        self.log_output.delete("1.0", tk.END)
