# Third-party notices

## Kronos

This repository includes the optional model runtime from
[Kronos](https://github.com/shiyu-coder/Kronos), **“A Foundation Model for the
Language of Financial Markets.”**

- Upstream commit reviewed/vendored: `67b630e67f6a18c9e9be918d9b4337c960db1e9a`
- Vendored paths: `third_party/kronos/model/`
- License: MIT; the upstream license text is preserved at
  `third_party/kronos/LICENSE`.
- Copyright: Copyright (c) 2025 ShiYu
- Model weights are not included in this repository. The optional adapter fetches
  open model weights from the `NeoQuasar` Hugging Face repositories at runtime.

Only the inference runtime is vendored. Kronos example GUIs, Chinese-market data,
training pipelines, generated forecast artifacts, and the upstream example
backtester were deliberately not copied because they do not fit this project's
NSE, risk-limited paper-trading architecture.
