## Reference PR: #1491 (EXAONE4 Support)

This PR shows the pattern for adding a simple transformer model:
- **Repository:** https://github.com/huggingface/optimum-intel
- **PR:** https://github.com/huggingface/optimum-intel/pull/1491
- **Title:** "Support exaone4 model"
- **Additions:** 47 lines


### Full Diff from Reference PR #1491

```diff
diff --git a/docs/source/openvino/models.mdx b/docs/source/openvino/models.mdx
index 21a4e2a202..17a9bcef2d 100644
--- a/docs/source/openvino/models.mdx
+++ b/docs/source/openvino/models.mdx
@@ -54,6 +54,7 @@ Here is the list of the supported architectures :
 - Encoder Decoder
 - ESM
 - Exaone
+- Exaone4
 - Falcon
 - Falcon-Mamba
 - Flaubert
```

```diff
diff --git a/optimum/exporters/openvino/model_configs.py b/optimum/exporters/openvino/model_configs.py
index cba7c547f0..531aa27a2a 100644
--- a/optimum/exporters/openvino/model_configs.py
+++ b/optimum/exporters/openvino/model_configs.py
@@ -605,6 +605,18 @@ class ExaoneOpenVINOConfig(LlamaOpenVINOConfig):
     pass


+@register_in_tasks_manager(
+    "exaone4",
+    *[
+        "text-generation",
+        "text-generation-with-past",
+    ],
+    library_name="transformers",
+)
+class Exaone4OpenVINOConfig(LlamaOpenVINOConfig):
+    MIN_TRANSFORMERS_VERSION = "4.54.0"
+
+
 @register_in_tasks_manager(
     "arcee",
     *[
```

```diff
diff --git a/tests/openvino/test_decoder.py b/tests/openvino/test_decoder.py
index ee9811d9ab..5573c9b5c8 100644
--- a/tests/openvino/test_decoder.py
+++ b/tests/openvino/test_decoder.py
@@ -128,6 +128,7 @@ class OVModelForCausalLMIntegrationTest(unittest.TestCase):

     if is_transformers_version(">=", "4.54.0"):
         # remote code models differs after transformers v4.54
+        SUPPORTED_ARCHITECTURES += ("exaone4",)
         SUPPORTED_ARCHITECTURES = tuple(set(SUPPORTED_ARCHITECTURES) - {"minicpm", "minicpm3", "arctic", "deepseek"})

     if is_transformers_version(">=", "4.55.0"):
@@ -151,6 +152,7 @@ class OVModelForCausalLMIntegrationTest(unittest.TestCase):
         "arctic",
         "chatglm4",
         "exaone",
+        "exaone4",
         "decilm",
         "minicpm3",
         "deepseek",
@@ -206,6 +208,7 @@ class OVModelForCausalLMIntegrationTest(unittest.TestCase):
         "phi3": 2,
         "gemma2": 4,
         "exaone": 8,
+        "exaone4": 1,
         "granite": 6,
         "granite-moe": 6,
         "glm": 28,
```

```diff
diff --git a/tests/openvino/test_export.py b/tests/openvino/test_export.py
index 53df9e5761..0ae988cce2 100644
--- a/tests/openvino/test_export.py
+++ b/tests/openvino/test_export.py
@@ -90,8 +90,10 @@ class ExportModelTest(unittest.TestCase):
     if is_transformers_version(">=", "4.49"):
         SUPPORTED_ARCHITECTURES.update({"zamba2": OVModelForCausalLM})

-    if is_transformers_version(">=", "4.54.0") and is_openvino_version(">=", "2025.4.0"):
-        SUPPORTED_ARCHITECTURES.update({"lfm2": OVModelForCausalLM})
+    if is_transformers_version(">=", "4.54"):
+        SUPPORTED_ARCHITECTURES.update({"exaone4": OVModelForCausalLM})
+        if is_openvino_version(">=", "2025.4.0"):
+            SUPPORTED_ARCHITECTURES.update({"lfm2": OVModelForCausalLM})

     EXPECTED_DIFFUSERS_SCALE_FACTORS = {
         "stable-diffusion-xl": {"vae_encoder": "128.0", "vae_decoder": "128.0"},
```

```diff
diff --git a/tests/openvino/test_exporters_cli.py b/tests/openvino/test_exporters_cli.py
index 68712a8d07..65031b1b77 100644
--- a/tests/openvino/test_exporters_cli.py
+++ b/tests/openvino/test_exporters_cli.py
@@ -120,6 +120,12 @@ class OVCLIExportTestCase(unittest.TestCase):
             ]
         )

+    if is_transformers_version(">=", "4.54"):
+        SUPPORTED_ARCHITECTURES.extend(
+            [
+                ("text-generation-with-past", "exaone4"),
+            ]
+        )
     if is_transformers_version(">=", "4.52.1") and is_openvino_version(">=", "2025.4.0"):
         SUPPORTED_ARCHITECTURES.extend(
             [
@@ -153,6 +159,7 @@ class OVCLIExportTestCase(unittest.TestCase):
         "falcon-mamba": 2,
         "qwen3": 2,
         "zamba2": 2,
+        "exaone4": 2,
         "bitnet": 2,
     }
```

```diff
diff --git a/tests/openvino/utils_tests.py b/tests/openvino/utils_tests.py
index 7bbed8ac2e..a43223ebc3 100644
--- a/tests/openvino/utils_tests.py
+++ b/tests/openvino/utils_tests.py
@@ -79,6 +79,7 @@
     "electra": "optimum-intel-internal-testing/tiny-random-electra",
     "esm": "optimum-intel-internal-testing/tiny-random-EsmModel",
     "exaone": "optimum-intel-internal-testing/tiny-random-exaone",
+    "exaone4": "optimum-intel-internal-testing/tiny-random-exaone4",
     "gemma": "optimum-intel-internal-testing/tiny-random-GemmaForCausalLM",
     "gemma2": "optimum-intel-internal-testing/tiny-random-gemma2",
     "got_ocr2": "optimum-intel-internal-testing/tiny-random-got-ocr2-hf",
@@ -346,6 +347,7 @@
         "resampler_model": 6,
     },
     "zamba2": {"model": 44},
+    "exaone4": {"model": 16},
     "lfm2": {"model": 52},
 }
```

