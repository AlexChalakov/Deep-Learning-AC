import time
import torch

def polynomial_fun(w, x):
    """
    Implement a polynomial function polynomial_fun, that takes two input arguments, 
    a weight vector 𝐰 of size 𝑀 + 1 
    and an input scalar variable 𝑥, 
    and returns the function value 𝑦.

    Input:
    - w: weight vector of size M + 1
    - x: input scalar variable
    
    Output:
    - y: evaluated polynomial function values
    """
    # ensuring x is a float tensor
    x = x.float()

    # [x^0, x^1, ..., x^M]
    powers_X = torch.pow(x.unsqueeze(-1), torch.arange(w.size(0), dtype=torch.float32, device=x.device))

    # using the dot product to get the sum of the polynomial
    y = torch.matmul(powers_X, w)
    return y

def fit_polynomial_ls(x, t, M):
    """
    Fitting a polynomial of degree M to data (x, t) using the least squares method

    Input:
    - x: input data points
    - t: target data points
    - M: degree of polynomial

    Output:
    - w: the optimized weight vector
    
    """
    # ensuring x and t are float tensors
    x = x.float()
    t = t.float()
    
    # creating a matrix of size (N, M+1) where each row is [x^0, x^1, ..., x^M]
    powers = torch.arange(M + 1).to(x.dtype)
    powers_X = torch.pow(x.unsqueeze(1), powers)
    
    # constructing the normal equation components X^T X and X^T t
    XtX = powers_X.t().matmul(powers_X)
    Xtt = powers_X.t().matmul(t.unsqueeze(1))
    
    # solve the linear system (X^T X)w = X^T t for weights w
    w = torch.linalg.solve(XtX, Xtt).squeeze()
    
    return w

def fit_polynomial_sgd(x,t,M, learning_rate, batch_size):
    """
    Implements stochastic gradient descent to fit polynomial functions, 
    incorporating the same parameters as fit_polynomial_ls with the addition of learning rate and batch size specifications.

    Input:
    - x: input data points
    - t: target data points
    - M: degree of polynomial
    - learning_rate: learning rate
    - batch_size: batch size

    Output:
    - w: the optimized weight vector using SGD
    """
    # ensuring x and t are float tensors
    x = x.float()
    t = t.float()

    # initialize weights randomly
    w = torch.randn(M + 1, requires_grad=True)

    # implementing the Loss function: Mean Squared Error - loss = ((y_pred - batch_t)**2).mean()
    loss_fn = torch.nn.MSELoss(reduction='sum')

    # implementing the Optimizer: Stochastic Gradient Descent
    optimizer = torch.optim.SGD([w], lr=learning_rate)

    # Training loop
    num_batches = x.size(0) // batch_size
    for batch in range(num_batches):
        # Sample a minibatch
        indices = torch.randperm(x.size(0))[:batch_size]
        x_batch, t_batch = x[indices], t[indices]

        # Compute predictions and loss
        y_pred = sum(w[i] * x_batch**i for i in range(M + 1))
        loss = loss_fn(y_pred, t_batch) / batch_size

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Report loss
        print(f"Batch {batch + 1}/{num_batches}, Loss: {loss.item()}")

    return w.detach()

def generate_data(w, num_samples, noise_std=0.5):
    """
    Generate data points by adding Gaussian noise to the output of the polynomial function.

    Input:
    - w: weight vector of size M + 1
    - num_samples: number of samples to generate
    - noise_std: standard deviation of the Gaussian noise

    Output:
    - x: input data points
    - t: target data points
    """
    # makes sure the values are spaced within the range -20 and 20
    x = torch.linspace(-20, 20, steps=num_samples)

    # calling the polynomial function to get the y values
    y = polynomial_fun(w, x)
    noise = torch.randn(y.shape) * noise_std
    t = y + noise
    return x, t

def main():
    # Polynomial coefficients and noise standard deviation
    w_true = torch.tensor([1, 2, 3], dtype=torch.float32)
    noise_std = 0.5

    # Generate training and test sets
    x_train, t_train = generate_data(w=w_true, num_samples=20, noise_std=noise_std)
    x_test, t_test = generate_data(w=w_true, num_samples=10, noise_std=noise_std)

    # Now calling fit_polynomial_ls to fit the polynomial function to the training data
    degrees = [2, 3, 4]

    for M in degrees:
        # Fit the polynomial to the training data
        w_opt = fit_polynomial_ls(x_train, t_train, M)

        # Predict on training and test sets
        y_train_pred = polynomial_fun(w_opt, x_train)
        y_test_pred = polynomial_fun(w_opt, x_test)

    # Calculate the true values without noise for comparison
    y_true_train = polynomial_fun(w_true.float(), x_train)
    y_true_test = polynomial_fun(w_true.float(), x_test)

    # a) Difference between observed training data and true curve
    diff_train_observed_true = t_train - y_true_train
    mean_diff_train_observed_true = diff_train_observed_true.mean().item()
    std_diff_train_observed_true = diff_train_observed_true.std().item()

    # b) Difference between LS-predicted values and true curve for training data
    diff_train_pred_true = y_train_pred - y_true_train
    mean_diff_train_pred_true = diff_train_pred_true.mean().item()
    std_diff_train_pred_true = diff_train_pred_true.std().item()

    #print(f"Degree M={M}:")
    print("\n--- Comparison ---")
    print(f"Observed vs. True on Training: Mean Diff = {mean_diff_train_observed_true:.6f}, StdDev = {std_diff_train_observed_true:.6f}")
    print(f"LS-Predicted vs. True on Training: Mean Diff = {mean_diff_train_pred_true:.6f}, StdDev = {std_diff_train_pred_true:.6f}")


    learning_rate = 0.01
    minibatch_size = 5  # Example value; adjust based on your dataset size and preference

    print("\n--- SGD-Based Fitting ---")
    for M in degrees:
        w_opt_sgd = fit_polynomial_sgd(x_train, t_train, M, learning_rate, minibatch_size)
        
        y_train_pred_sgd = polynomial_fun(w_opt_sgd, x_train)
        y_test_pred_sgd = polynomial_fun(w_opt_sgd, x_test)

        # Difference between SGD-predicted values and true curve for training data
        diff_sgd_true = y_train_pred_sgd - polynomial_fun(w_true.float(), x_train)
        mean_diff_sgd_true = diff_sgd_true.mean().item()
        std_diff_sgd_true = diff_sgd_true.std().item()

        print(f"Degree M={M}:")
        print(f"SGD-Predicted vs. True on Training: Mean Diff = {mean_diff_sgd_true:.6f}, StdDev = {std_diff_sgd_true:.6f}")

    print("\n--- Accuracy Comparison (RMSE) ---")
    for M in degrees:
        # Assuming w_opt and w_opt_sgd are available from the previous steps
        y_test_pred_ls = polynomial_fun(w_opt, x_test)
        y_test_pred_sgd = polynomial_fun(w_opt_sgd, x_test)

        rmse_ls = torch.sqrt(torch.mean((y_test_pred_ls - polynomial_fun(w_true.float(), x_test))**2))
        rmse_sgd = torch.sqrt(torch.mean((y_test_pred_sgd - polynomial_fun(w_true.float(), x_test))**2))

        print(f"Degree M={M}: LS RMSE = {rmse_ls:.6f}, SGD RMSE = {rmse_sgd:.6f}")

    print("\n--- Speed Comparison ---")
    for M in degrees:
        start_time_ls = time.time()
        w_opt = fit_polynomial_ls(x_train, t_train, M)
        time_ls = time.time() - start_time_ls

        start_time_sgd = time.time()
        w_opt_sgd = fit_polynomial_sgd(x_train, t_train, M, learning_rate, minibatch_size)
        time_sgd = time.time() - start_time_sgd

        print(f"Degree M={M}: LS Time = {time_ls:.6f}s, SGD Time = {time_sgd:.6f}s")

if __name__ == "__main__":
    main()
