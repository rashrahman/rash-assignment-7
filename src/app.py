from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
import os
from scipy.stats import t

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with your own secret key, needed for session management


def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots

    # TODO 1: Generate a random dataset X of size N with values between 0 and 1
    X = np.random.rand(N)

    # TODO 2: Generate a random dataset Y using the specified beta0, beta1, mu, and sigma2
    # Y = beta0 + beta1 * X + mu + error term
    error = np.random.normal(mu, np.sqrt(sigma2), N)
    Y = beta0 + beta1 * X + error

    # TODO 3: Fit a linear regression model to X and Y
    X_reshape = X.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X_reshape, Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # TODO 4: Generate a scatter plot of (X, Y) with the fitted regression line
    Y_pred = model.predict(X_reshape)
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, alpha=0.5, color='blue', label='Data points')
    plt.plot(X, Y_pred, 'r-', label='Regression line')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Linear Regression: y = {slope:.2f}x + {intercept:.2f}')
    plt.legend()
    plt.grid(True)

    # Create the static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)

    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()

    # TODO 5: Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []

    for _ in range(S):
        # TODO 6: Generate simulated datasets using the same beta0 and beta1
        X_sim = np.random.rand(N)
        error_sim = np.random.normal(mu, np.sqrt(sigma2), N)
        Y_sim = beta0 + beta1 * X_sim + error_sim

        # TODO 7: Fit linear regression to simulated data and store slope and intercept
        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)
        sim_slope = sim_model.coef_[0]
        sim_intercept = sim_model.intercept_

        slopes.append(sim_slope)
        intercepts.append(sim_intercept)

    # TODO 8: Plot histograms of slopes and intercepts
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Observed Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Observed Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    # TODO 9: Return data needed for further analysis, including slopes and intercepts
    # Calculate proportions of slopes and intercepts more extreme than observed
    slope_more_extreme = np.mean(np.abs(np.array(slopes) - beta1) >= np.abs(slope - beta1))
    intercept_more_extreme = np.mean(np.abs(np.array(intercepts) - beta0) >= np.abs(intercept - beta0))

    # Return data needed for further analysis
    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_more_extreme,
        slopes,
        intercepts,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        # Return render_template with variables
        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    # This route handles data generation (same as above)
    return index()


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    # TODO 10: Calculate p-value based on test type
    if test_type == '>':
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == '<':
        p_value = np.mean(simulated_stats <= observed_stat)
    elif test_type == '≠':
        p_value = np.mean(np.abs(simulated_stats - hypothesized_value) >= np.abs(observed_stat - hypothesized_value))
    else:
        p_value = None

    # TODO 11: If p_value is very small (e.g., <= 0.0001), set fun_message to a fun message
    if p_value is not None and p_value <= 0.0001:
        fun_message = "Wow! You've encountered a rare event!"
    else:
        fun_message = None

    # TODO 12: Plot histogram of simulated statistics
    plot3_path = "static/plot3.png"
    plt.figure(figsize=(10, 6))
    plt.hist(simulated_stats, bins=30, alpha=0.7, label='Simulated Statistics')
    plt.axvline(observed_stat, color='red', linestyle='--', linewidth=2, label=f'Observed Statistic: {observed_stat:.4f}')
    plt.axvline(hypothesized_value, color='green', linestyle='-', linewidth=2, label=f'Hypothesized Value: {hypothesized_value:.4f}')
    plt.xlabel('Statistic Value')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Simulated {parameter.capitalize()}s')
    plt.legend()
    plt.savefig(plot3_path)
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        p_value=p_value,
        fun_message=fun_message,
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    X = np.array(session.get("X"))
    Y = np.array(session.get("Y"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = slope
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = intercept
        true_param = beta0

    # TODO 14: Calculate mean and standard deviation of the estimates
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)

    # TODO 15: Calculate confidence interval for the parameter estimate
    # Use the t-distribution and confidence_level
    from scipy import stats
    alpha = 1 - confidence_level/100
    t_value = stats.t.ppf(1 - alpha/2, S-1)
    margin_error = t_value * std_estimate / np.sqrt(S)
    ci_lower = mean_estimate - margin_error
    ci_upper = mean_estimate + margin_error

    # TODO 16: Check if confidence interval includes true parameter
    includes_true = ci_lower <= true_param <= ci_upper

    # TODO 17: Plot the individual estimates as gray points and confidence interval
    # Plot the mean estimate as a colored point which changes if the true parameter is included
    # Plot the confidence interval as a horizontal line
    # Plot the true parameter value
    plot4_path = "static/plot4.png"
    plt.figure(figsize=(10, 6))
    plt.scatter(estimates, [0]*len(estimates), alpha=0.3, color='gray', label='Simulated Estimates')
    plt.scatter(mean_estimate, 0, color='blue' if includes_true else 'red', s=100, label='Mean Estimate')
    plt.hlines(0, ci_lower, ci_upper, color='blue' if includes_true else 'red', linewidth=2, label=f'{confidence_level}% CI')
    plt.axvline(true_param, color='green', linestyle='--', label=f'True {parameter}')
    plt.xlabel(f'{parameter.capitalize()} Estimate')
    plt.title(f'{confidence_level}% Confidence Interval for {parameter.capitalize()} (Mean Estimate)')
    plt.legend()
    plt.savefig(plot4_path)
    plt.close()
    # Write code here to generate and save the plot

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )



if __name__ == "__main__":
    app.run(debug=True)
