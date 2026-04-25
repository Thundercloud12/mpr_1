from typing import Callable

import tkinter as tk
from tkinter import ttk

from algorithms.benchmarks import PRESET_CONFIG, PRESET_NAMES


class ExperimentSetupSection:
    def __init__(
        self,
        parent: ttk.Frame,
        on_initialize: Callable[[], None],
        on_step: Callable[[], None],
        on_run: Callable[[], None],
        on_preset_changed: Callable[[], None] | None,
        default_population: str,
        default_iterations: str,
        default_custom_expression: str,
    ) -> None:
        self._on_preset_changed_callback = on_preset_changed
        self.default_population = default_population
        self.default_iterations = default_iterations
        self.default_custom_expression = default_custom_expression

        controls = ttk.LabelFrame(parent, text="Experiment Setup", padding=10)
        controls.pack(fill=tk.X, pady=(0, 10))
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Preset:", style="Body.TLabel").grid(
            row=0, column=0, sticky="w", pady=4
        )
        self.preset_var = tk.StringVar(value="Sphere")
        self.preset_combo = ttk.Combobox(
            controls,
            textvariable=self.preset_var,
            values=PRESET_NAMES,
            width=24,
            state="readonly",
        )
        self.preset_combo.grid(row=0, column=1, sticky="w", pady=4)
        self.preset_combo.bind("<<ComboboxSelected>>", self.on_preset_changed)

        ttk.Label(controls, text="Objective Function f(x):", style="Body.TLabel").grid(
            row=1, column=0, sticky="w", pady=4
        )
        self.func_entry = ttk.Entry(controls, width=40)
        self.func_entry.grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(controls, text="Dimensions:", style="Body.TLabel").grid(
            row=2, column=0, sticky="w", pady=4
        )
        self.dim_entry = ttk.Entry(controls, width=12)
        self.dim_entry.grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(controls, text="Bounds (min,max;...):", style="Body.TLabel").grid(
            row=3, column=0, sticky="w", pady=4
        )
        self.bounds_entry = ttk.Entry(controls, width=40)
        self.bounds_entry.grid(row=3, column=1, sticky="ew", pady=4)

        ttk.Label(controls, text="Population Size:", style="Body.TLabel").grid(
            row=4, column=0, sticky="w", pady=4
        )
        self.pop_entry = ttk.Entry(controls, width=12)
        self.pop_entry.grid(row=4, column=1, sticky="w", pady=4)

        ttk.Label(controls, text="Iterations:", style="Body.TLabel").grid(
            row=5, column=0, sticky="w", pady=4
        )
        self.iter_entry = ttk.Entry(controls, width=12)
        self.iter_entry.grid(row=5, column=1, sticky="w", pady=4)

        run_buttons = ttk.Frame(controls)
        run_buttons.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(10, 4))
        ttk.Button(
            run_buttons,
            text="Initialize",
            command=on_initialize,
            style="Action.TButton",
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            run_buttons,
            text="Step",
            command=on_step,
            style="Action.TButton",
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(
            run_buttons,
            text="Run",
            command=on_run,
            style="Action.TButton",
        ).pack(side=tk.LEFT, padx=6)

        self.on_preset_changed()

    def _set_entry_if_empty(self, entry_widget: ttk.Entry, value: str) -> None:
        if not entry_widget.get().strip():
            entry_widget.insert(0, value)

    def _replace_entry_text(self, entry_widget: ttk.Entry, value: str) -> None:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, value)

    def on_preset_changed(self, _event=None) -> None:
        preset = self.preset_var.get()
        self.func_entry.config(state="normal")

        if preset == "Custom":
            self._set_entry_if_empty(self.func_entry, self.default_custom_expression)
            self._set_entry_if_empty(self.dim_entry, "2")
            self._set_entry_if_empty(self.bounds_entry, "-5.12,5.12;-5.12,5.12")
        else:
            cfg = PRESET_CONFIG[preset]
            self._replace_entry_text(self.func_entry, cfg.expression)
            self.func_entry.config(state="disabled")
            self._replace_entry_text(self.dim_entry, str(cfg.dim))
            self._replace_entry_text(self.bounds_entry, cfg.bounds)

        self._set_entry_if_empty(self.pop_entry, self.default_population)
        self._set_entry_if_empty(self.iter_entry, self.default_iterations)

        if self._on_preset_changed_callback is not None:
            self._on_preset_changed_callback()

    def get_values(self) -> dict[str, str]:
        return {
            "preset": self.preset_var.get(),
            "function_expression": self.func_entry.get().strip(),
            "dim": self.dim_entry.get().strip(),
            "bounds": self.bounds_entry.get().strip(),
            "population_size": self.pop_entry.get().strip(),
            "iterations": self.iter_entry.get().strip(),
        }
