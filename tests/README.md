# Testing Guidelines

## Philosophy: Test Core Requirements Only

**Focus on critical paths that validate core functionality.** Avoid over-testing edge cases, implementation details, or redundant scenarios. Keep tests fast, focused, and maintainable.

---

## Best Practices

### 1. **Test Core Requirements Only**

✅ **DO Test:**
- Critical business logic (transaction parsing, address checksumming)
- Error handling for known failure modes (invalid URLs, missing files)
- Core algorithm correctness (retry logic, batch processing)
- Data validation (format checking, type conversions)

❌ **DON'T Test:**
- Implementation details (internal helper methods unless critical)
- Framework behavior (e.g., testing that `json.dump()` works)
- Multiple variations of the same logic path
- Scenarios that are "nice to have" but not MVP requirements

**Example:**
```python
# ✅ GOOD - Tests core requirement
def test_placeholder_url_rejected():
    """Reject placeholder URLs - critical for production safety."""
    with pytest.raises(ValueError, match="Invalid RPC URL"):
        Web3ConnectionManager(rpc_url="PLACEHOLDER_RPC_URL")

# ❌ BAD - Over-testing edge cases
def test_url_with_trailing_slash():
    """Test URL normalization with trailing slash."""
    # Not critical - implementation detail
```

---

### 2. **Keep Tests Fast**

**Target: <5 seconds total test suite runtime**

✅ **DO:**
- Use fast backoff values in tests: `backoff_factor=0.01` instead of `2.0`
- Mock network calls, file I/O, and external dependencies
- Use `tmp_path` fixture for file operations (cleaned up automatically)
- Avoid `time.sleep()` unless absolutely necessary

❌ **DON'T:**
- Test actual timing behavior (e.g., "verify backoff takes 3 seconds")
- Make real network requests to RPC endpoints
- Create large test datasets
- Run integration tests in unit test suite

**Example:**
```python
# ✅ GOOD - Fast test with minimal backoff
def test_retry_logic_succeeds_after_failures(self, mock_web3):
    manager = Web3ConnectionManager(
        rpc_url="http://localhost:8545",
        max_retries=3,
        backoff_factor=0.01,  # Fast for testing
    )
    # Test completes in milliseconds

# ❌ BAD - Slow test that actually waits
def test_retry_with_exponential_backoff_timing(self):
    manager = Web3ConnectionManager(backoff_factor=2.0)
    start_time = time.time()
    # ... test that takes 3+ seconds
```

---

### 3. **Write Concise Tests**

**Keep each test function under 15 lines.** If a test is longer, it's probably testing too much.

✅ **DO:**
- Test one thing per test function
- Use descriptive names: `test_<what>_<expected_behavior>`
- Keep setup minimal (use fixtures)
- Assert on specific, meaningful outcomes

❌ **DON'T:**
- Test multiple scenarios in one function
- Write verbose setup code repeatedly
- Assert on every field unless critical
- Add comments explaining what the test does (name should be clear)

**Example:**
```python
# ✅ GOOD - Concise, single purpose
def test_load_hashes_skips_invalid_lines(self, tmp_path):
    """Invalid hashes and comments should be skipped."""
    hash_file = tmp_path / "hashes.txt"
    with open(hash_file, "w") as f:
        f.write("# Comment\n0x1234\n")  # Too short
        f.write("0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef\n")
    
    hashes = load_transaction_hashes(hash_file)
    assert len(hashes) == 1

# ❌ BAD - Testing too many things
def test_comprehensive_hash_validation(self):
    # Tests comments, empty lines, invalid length, invalid hex,
    # with/without 0x prefix, Unicode characters, etc.
    # 50+ lines of test code...
```

---

### 4. **Use Fixtures Wisely**

✅ **DO:**
- Create fixtures for commonly used test data
- Use `pytest.fixture` for shared setup
- Keep fixtures simple and reusable
- Store complex fixtures in `tests/fixtures/` directory

❌ **DON'T:**
- Create fixtures for one-off data
- Make fixtures that do too much
- Nest fixtures more than 2 levels deep

**Example:**
```python
# ✅ GOOD - Simple, reusable fixture
@pytest.fixture
def sample_tx_hashes_file(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_tx_hashes.txt"

# ❌ BAD - Over-complicated fixture
@pytest.fixture
def fully_mocked_extraction_pipeline(mock_web3, mock_config, tmp_path):
    # 30+ lines of complex setup...
```

---

### 5. **Mock External Dependencies**

✅ **DO:**
- Mock network calls (RPC endpoints, APIs)
- Mock file system operations when possible
- Use `unittest.mock.patch` for external services
- Mock at the boundary (e.g., `Web3()` class, not internal methods)

❌ **DON'T:**
- Mock internal methods you own (test them directly)
- Over-mock to the point tests become meaningless
- Mock simple Python operations (list comprehensions, etc.)

**Example:**
```python
# ✅ GOOD - Mock at the boundary
with patch("scripts.extraction.utils.Web3") as mock_web3_class:
    mock_web3_class.return_value.is_connected.return_value = True
    manager = Web3ConnectionManager(rpc_url="http://localhost:8545")

# ❌ BAD - Mocking internal implementation
with patch.object(manager, '_serialize_transaction'):
    # Testing nothing useful
```

---

### 6. **Avoid Redundant Tests**

If two tests cover the same code path with minor variations, keep only one.

✅ **DO:**
- Combine similar test cases using `@pytest.mark.parametrize`
- Test the most important variation
- Focus on boundary conditions that matter

❌ **DON'T:**
- Test every possible input variation
- Write tests for "symmetry" (e.g., if you test success, you don't need 5 failure tests)
- Copy-paste tests with minor changes

---

## Test Organization

### Directory Structure
```
tests/
├── README.md                  # This file
├── test_extraction.py         # Core extraction tests
├── test_decoders.py          # Transaction decoder tests
├── test_dataset.py           # Dataset preparation tests
└── fixtures/                 # Shared test data
    ├── sample_transactions.json
    └── sample_tx_hashes.txt
```

### Test File Template
```python
"""
Brief description of what module/feature is being tested.

Focus: Core functionality only
- Critical path 1
- Critical path 2
- Critical path 3
"""

import pytest
from your.module import function_to_test


@pytest.fixture
def shared_test_data():
    """Describe fixture purpose."""
    return {"key": "value"}


class TestFeatureName:
    """Test core <feature> functionality."""
    
    def test_happy_path(self, shared_test_data):
        """Test successful execution."""
        result = function_to_test(shared_test_data)
        assert result is not None
    
    def test_critical_error_case(self):
        """Test critical failure mode."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```

---

## Running Tests

### Quick Test Run
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_extraction.py -v

# Run specific test
pytest tests/test_extraction.py::TestWeb3Connection::test_retry_logic -v
```

### With Coverage
```bash
# Generate coverage report
pytest tests/ --cov=scripts --cov-report=term-missing

# Target: >70% coverage on critical modules
# Don't chase 100% - focus on core paths
```

### Fast Development Workflow
```bash
# Run tests on file change (requires pytest-watch)
ptw tests/ -- -v

# Run only failed tests from last run
pytest --lf -v
```

---

## Current Test Metrics

**Test Count:** 8 tests  
**Runtime:** ~3.5 seconds  
**Coverage:** 50-70% (focused on core modules)

These metrics represent **optimal balance** between coverage and maintainability for this project phase.

---

## When to Add Tests

✅ **Add tests when:**
- Implementing new core functionality
- Fixing a critical bug (regression test)
- Adding a new decoder or pipeline stage
- Refactoring breaks existing tests

❌ **Don't add tests for:**
- Minor refactors that don't change behavior
- "Just in case" scenarios not in requirements
- Every possible input combination
- Implementation details that may change

---

## Testing Anti-Patterns to Avoid

### ❌ Testing Implementation Details
```python
# BAD - Breaks when you refactor internal method names
def test_internal_helper_method():
    assert manager._internal_method() == expected
```

### ❌ Testing Constants
```python
# BAD - Not testing logic, just definitions
def test_default_timeout_is_30():
    assert DEFAULT_TIMEOUT == 30
```

### ❌ Meaningless Assertions
```python
# BAD - Assert doesn't validate anything useful
def test_function_returns_something():
    result = function()
    assert result is not None  # So what?
```

### ❌ Brittle String Matching
```python
# BAD - Breaks when error message wording changes
with pytest.raises(ValueError, match="Invalid RPC URL provided to connection manager"):
    # Exact string match is fragile
```

---

## Summary: Testing Checklist

Before adding a test, ask:
- [ ] Does this test a **core requirement** from SPEC.md?
- [ ] Is this the **simplest test** that validates the behavior?
- [ ] Does it run in **<1 second**?
- [ ] Will it **fail if the feature breaks**?
- [ ] Is it **not redundant** with existing tests?

If you answered **"yes"** to all 5, add the test. Otherwise, reconsider.

**Remember: Quality > Quantity. Focus on critical paths, keep tests fast, and avoid over-engineering.**
