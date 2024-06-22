# public_tests.py
import numpy as np
def compute_cost_test(compute_cost_fn):
    # Test data
    x_test = np.array([1, 2, 3])
    y_test = np.array([2, 2.5, 3.5])
    w_test = 1
    b_test = 1

    # Expected result
    expected_cost = 0.08333333333333333

    # Compute the cost using the provided function
    computed_cost = compute_cost_fn(x_test, y_test, w_test, b_test)

    # Compare the computed cost with the expected cost
    assert np.isclose(computed_cost, expected_cost), f"Expected {expected_cost}, but got {computed_cost}"

    print("compute_cost_test passed!")

# Save this file in the same directory as your script.


def compute_gradient_test(compute_gradient_fn):
    # Test data
    x_test = np.array([1, 2, 3])
    y_test = np.array([2, 2.5, 3.5])
    w_test = 1
    b_test = 1

    # Expected gradients
    expected_dj_dw = 0.8333333333333334
    expected_dj_db = 0.3333333333333333

    # Compute the gradients using the provided function
    dj_dw, dj_db = compute_gradient_fn(x_test, y_test, w_test, b_test)

    # Compare the computed gradients with the expected gradients
    assert np.isclose(dj_dw, expected_dj_dw), f"Expected dj_dw: {expected_dj_dw}, but got {dj_dw}"
    assert np.isclose(dj_db, expected_dj_db), f"Expected dj_db: {expected_dj_db}, but got {dj_db}"

    print("compute_gradient_test passed!")
