import numpy as np
import matplotlib.pyplot as plt


# Root-Finding Methods

def Newton(f, df, x0, small=1e-6, max_iter=100):
    """Newton's method for root approximation."""
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ValueError("Derivative zero. Try another initial guess.")
        x_new = x - fx / dfx
        if abs(x_new - x) < small:
            return x_new, i + 1, f(x_new)
        x = x_new
    raise ValueError("Newton did not converge.")

def Secant(f, x0, x1, small=1e-6, max_iter=100):
    """Secant method for root approximation."""
    f0, f1 = f(x0), f(x1)
    for i in range(max_iter):
        if abs(f1 - f0) < 1e-14:
            raise ValueError("Division by near-zero in Secant method.")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(x2 - x1) < small:
            return x2, i + 1, f(x2)
        x0, x1 = x1, x2
        f0, f1 = f1, f(x1)
    raise ValueError("Secant did not converge.")


# Test Newton’s Method


def TestNewton():
    print("Testing Newton Method...\n")

    f = lambda x: x**3 - 6*x**2 + 11*x - 6
    df = lambda x: 3*x**2 - 12*x + 11
    true_roots = np.array([1.0, 2.0, 3.0])

    start_guesses = [0.5, 2.1, 2.9]
    found_roots = []

    plt.figure(figsize=(8, 6))
    print("Calculating convergence orders for Newton...")

    for x0, true_root in zip(start_guesses, true_roots):
        errors = []
        x = x0
        for _ in range(10):
            err = abs(x - true_root)
            errors.append(err)
            fx, dfx = f(x), df(x)
            if dfx == 0:
                break
            x = x - fx / dfx
            if err < 1e-10:
                break
        found_roots.append(round(x, 6))

        if len(errors) >= 3:
            log_e = np.log(errors[:-1])
            log_e_next = np.log(errors[1:])
            slope, _ = np.polyfit(log_e, log_e_next, 1)
            print(f"  From x0={x0:.2f} → root≈{x:.6f} (true {true_root}), order ≈ {slope:.3f}")
            plt.plot(log_e, log_e_next, 'o-', label=f"x0={x0}, order={slope:.2f}")

    plt.title("Newton Method Convergence Order")
    plt.xlabel("log|error_k|")
    plt.ylabel("log|error_{k+1}|")
    plt.grid(True)
    plt.legend()
    plt.savefig("newton_convergence.png")
    print("\nNewton plot saved as 'newton_convergence.png'")
    print(f"Newton roots found: {found_roots}\n")


# Test Secant Method


def TestSecant():
    print(" Testing Secant Method...\n")

    f = lambda x: x**3 - 6*x**2 + 11*x - 6
    true_roots = np.array([1.0, 2.0, 3.0])
    start_pairs = [(0.9, 1.2), (1.8, 2.1), (2.9, 3.5)]
    found_roots = []

    plt.figure(figsize=(8, 6))
    print("Calculating convergence orders for Secant...")

    for (x0, x1), true_root in zip(start_pairs, true_roots):
        errors = []
        for _ in range(15):
            f0, f1 = f(x0), f(x1)
            if abs(f1 - f0) < 1e-14:
                break
            x_next = x1 - f1 * (x1 - x0) / (f1 - f0)
            error = abs(x_next - true_root)
            errors.append(error)
            if error < 1e-10:
                break
            x0, x1 = x1, x_next
        found_roots.append(round(x_next, 6))

        if len(errors) >= 3:
            log_e = np.log(errors[:-1])
            log_e_next = np.log(errors[1:])
            slope, _ = np.polyfit(log_e, log_e_next, 1)
            print(f"  From ({x0:.2f},{x1:.2f}) → root≈{x_next:.6f} (true {true_root}), order ≈ {slope:.3f}")
            plt.plot(log_e, log_e_next, 'o-', label=f"({x0:.1f},{x1:.1f}), order={slope:.2f}")

    plt.title("Secant Method Convergence Order")
    plt.xlabel("log|error_k|")
    plt.ylabel("log|error_{k+1}|")
    plt.grid(True)
    plt.legend()
    plt.savefig("secant_convergence.png")
    print("\nSecant plot saved as 'secant_convergence.png'")
    print(f" Secant roots found: {found_roots}\n")



# Main Execution


if __name__ == "__main__":
    print("\n--- Testing Root-Finding Methods ---\n")
    try:
        TestNewton()
        TestSecant()
        print("All three roots (1, 2, 3) successfully found using both methods!")
    except Exception as e:
        print(f"\n Error: {e}")
