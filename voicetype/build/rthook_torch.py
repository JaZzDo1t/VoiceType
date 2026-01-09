"""
Runtime hook for torch to ensure proper initialization order in frozen builds.

This hook runs BEFORE main.py and ensures torch is fully imported
before any application code tries to use it.

The critical fix here is patching RecursiveScriptModule._construct which is needed
for torch.jit.load() and torch.package to work properly in frozen builds.
"""
import sys
import os

# Only run in frozen builds
if getattr(sys, 'frozen', False):
    _success = False
    _error = None

    # NOTE: Do NOT set PYTORCH_JIT=0 - it breaks TorchScript model loading!
    # The PYTORCH_JIT environment variable controls JIT compilation,
    # and setting it to 0 can cause issues with torch.jit.load()

    try:
        # Step 1: Import torch base
        import torch

        # Step 2: Force core C extensions to load
        _ = torch.Tensor
        _ = torch.zeros(1)

        # Step 3: Import torch.jit modules
        import torch.jit
        import torch.jit._script
        import torch.jit._recursive

        # Step 4: Critical fix - patch RecursiveScriptModule._construct if missing
        from torch.jit._script import RecursiveScriptModule

        if not hasattr(RecursiveScriptModule, '_construct'):
            print("[rthook_torch] RecursiveScriptModule._construct is missing, patching...", file=sys.stderr)

            # Define the _construct classmethod (complete implementation from PyTorch)
            @classmethod
            def _construct(cls, cpp_module, init_fn):
                """
                Construct a ScriptModule that's already initialized from C++.
                This must initialize all the internal dictionaries that Module expects.
                """
                instance = cls.__new__(cls)

                # Initialize internal dictionaries using object.__setattr__
                # to bypass Module's custom __setattr__ which requires these to exist
                object.__setattr__(instance, 'training', True)
                object.__setattr__(instance, '_parameters', {})
                object.__setattr__(instance, '_buffers', {})
                object.__setattr__(instance, '_modules', {})
                object.__setattr__(instance, '_backward_hooks', {})
                object.__setattr__(instance, '_backward_pre_hooks', {})
                object.__setattr__(instance, '_forward_hooks', {})
                object.__setattr__(instance, '_forward_pre_hooks', {})
                object.__setattr__(instance, '_forward_hooks_with_kwargs', {})
                object.__setattr__(instance, '_forward_pre_hooks_with_kwargs', {})
                object.__setattr__(instance, '_forward_hooks_always_called', {})
                object.__setattr__(instance, '_state_dict_hooks', {})
                object.__setattr__(instance, '_state_dict_pre_hooks', {})
                object.__setattr__(instance, '_load_state_dict_pre_hooks', {})
                object.__setattr__(instance, '_load_state_dict_post_hooks', {})
                object.__setattr__(instance, '_non_persistent_buffers_set', set())
                object.__setattr__(instance, '_is_full_backward_hook', None)

                # Set the C++ module - this is where forward() lives
                object.__setattr__(instance, '_c', cpp_module)
                object.__setattr__(instance, '_init_fn', init_fn)

                # Call the init function to set up submodules
                init_fn(instance)

                return instance

            # Patch the class
            RecursiveScriptModule._construct = _construct

            # Also patch torch.jit.RecursiveScriptModule reference
            torch.jit.RecursiveScriptModule = RecursiveScriptModule

            print("[rthook_torch] RecursiveScriptModule._construct patched successfully", file=sys.stderr)
        else:
            print("[rthook_torch] RecursiveScriptModule._construct exists", file=sys.stderr)

        # Step 5: Import other needed modules
        import torch.nn
        import torch.nn.modules
        import torch.nn.functional
        import torch.serialization

        # Step 6: Import torch.package modules
        import torch.package
        import torch.package.package_importer

        # Step 7: Verify basic functionality
        _ = torch.__version__
        _ = torch.nn.Module

        # Step 8: Test JIT model loading capability
        print("[rthook_torch] Testing JIT model loading capability...", file=sys.stderr)
        try:
            # Create a simple scripted function and verify it works
            @torch.jit.script
            def _test_fn(x: torch.Tensor) -> torch.Tensor:
                return x + 1

            test_result = _test_fn(torch.zeros(1))
            print(f"[rthook_torch] JIT scripting works: {test_result.item()}", file=sys.stderr)
        except Exception as jit_err:
            print(f"[rthook_torch] JIT scripting test failed: {jit_err}", file=sys.stderr)
            # Continue anyway, this is just a diagnostic

        _success = True
        print(f"[rthook_torch] PyTorch {torch.__version__} initialized successfully", file=sys.stderr)

    except Exception as e:
        _error = e
        import traceback
        print(f"[rthook_torch] PyTorch init failed: {type(e).__name__}: {e}", file=sys.stderr)
        print(f"[rthook_torch] Traceback: {traceback.format_exc()}", file=sys.stderr)

        # Try minimal fallback
        try:
            if 'torch' not in sys.modules:
                import torch
            _success = 'torch' in sys.modules and hasattr(sys.modules['torch'], '__version__')
            if _success:
                print(f"[rthook_torch] Fallback: torch base loaded", file=sys.stderr)
        except:
            pass

    # Set markers for app code
    sys._torch_rthook_success = _success
    sys._torch_rthook_error = _error
