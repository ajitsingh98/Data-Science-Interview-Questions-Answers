# Data Science Interview Questions And Answers

## Time Series Analysis and Forecasting

Contents
----

- [Introduction](#introduction)
- [Time Series Graphics](#time-series-graphics)
- [Time Series Decomposition](#time-series-decomposition)
- [Benchmark Forecasting Methods](#benchmark-forecasting-methods)
- [Time Series Regression Models](#time-series-regression-models)
- [Exponential Smoothing](#exponential-smoothing)
- [ARIMA Models](#arima-models)
- [Prophet Model](#prophet-model)
- [Vector Autoregressions](#vector-autoregressions)
- [Neural Networks](#neural-networks)

---

## Introduction

Q. What is forecasting?

<details><summary><b>Answer</b></summary>

Forecasting is about predicting the future as accurately as possible, given all of the information available, including historical data and knowledge of any future events that might impact the forecasts.

</details>

---

Q. What do you mean by time series?

<details><summary><b>Answer</b></summary>

Anything that is observed sequentially over time is a time series. The observations can be at regular intervals of time (e.g. hourly, daily, monthly etc.) or irregular intervals.

</details>

---

Q. Define following terms:
- Short-term forecasts
- Medium-term forecasts
- Long-term forecasts

<details><summary><b>Answer</b></summary>

*Short-term forecasts*

It is needed for the scheduling of personnel, production and transportation. As part of the scheduling process, forecasts of demand are often also required.

*Medium-term forecasts*

It is needed to determine future resource requirements, in order to purchase raw materials, hire personnel, or buy machinery and equipment.

*Long-term forecasts*

It is used in strategic planning. Such decisions must take account of market opportunities, environmental factors and internal resources.


</details>

---

Q. List down factors on which predictability of an event or quantity depends?

<details><summary><b>Answer</b></summary>

The predictability of an event or a quantity depends on several factors including
- how well we understand the factors that contribute to it
- how much data is available 
- how similar the future is to the past
- whether the forecasts can affect the thing we are trying to forecast

</details>

---

Q. Is it correct to assume that forecasts are not possible in a changing environment?

<details><summary><b>Answer</b></summary>

No it is not correct. Forecasts rarely assume that the environment is unchanging. What is normally assumed is that the way in which the environment is changing will continue into the future.

</details>

---

Q. How would you approach forecasting if there is no available data, or if the data you have is not relevant to the forecasts?

<details><summary><b>Answer</b></summary>

In this scenario we can use qualitative forecasting methods. These methods are not purely guesswork—there are well-developed structured approaches to obtaining good forecasts without using historical data. 

</details>

---

Q. When can we use Quantitative methods for forecasting use-cases?

<details><summary><b>Answer</b></summary>

Quantitative forecasting can be applied when two conditions are satisfied:

- Numerical information about the past is available
- It is reasonable to assume that some aspects of the past patterns will continue into the future

</details>

---



## Time Series Graphics

Q. Which is the most common plot in time series EDA?

<details><summary><b>Answer</b></summary>

Time plot - the observations are plotted against the time of observation, with consecutive observations joined by straight lines.

<table align='center'>
  <tr>
    <td align="center">
      <img src="img/time-plot.png" alt= "Time Plot" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> Time Plot </td>
  </tr>
</table>

</details>

---

Q. What is the difference between seasonal plot and time plot?

<details><summary><b>Answer</b></summary>

A seasonal plot is similar to a time plot except that the data are plotted against individual seasons in which the data were observed.

<table align='center'>
  <tr>
    <td align="center">
      <img src="img/seasonal_plot.png" alt= "Seasonal Plot" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> Seasonal Plot </td>
  </tr>
</table>

</details>

---

Q. What is the difference between seasonal plot and time plot?

<details><summary><b>Answer</b></summary>

A seasonal plot is similar to a time plot except that the data are plotted against individual seasons in which the data were observed.

<table align='center'>
  <tr>
    <td align="center">
      <img src="img/seasonal_plot.png" alt= "Seasonal Plot" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> Seasonal Plot </td>
  </tr>
</table>

</details>

---

Q. What is the benefit of seasonal plots?

<details><summary><b>Answer</b></summary>

A seasonal plot allows the underlying seasonal pattern to be seen more clearly, and is especially useful in identifying years in which the pattern changes.

</details>

---

Q. What is seasonal subseries plots?

<details><summary><b>Answer</b></summary>

An alternative plot that emphasises the seasonal patterns is where the data for each season are collected together in separate mini time plots.

<table align='center'>
  <tr>
    <td align="center">
      <img src="img/seasonal_subseries_plots.png" alt= "Seasonal Subseries Plots" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> Seasonal Subseries Plots </td>
  </tr>
</table>

</details>

---

Q. What is seasonal subseries plots?

<details><summary><b>Answer</b></summary>

An alternative plot that emphasises the seasonal patterns is where the data for each season are collected together in separate mini time plots.

<table align='center'>
  <tr>
    <td align="center">
      <img src="img/seasonal_subseries_plots.png" alt= "Seasonal Subseries Plots" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> Seasonal Subseries Plots </td>
  </tr>
</table>

</details>

---

Q. Can we use scatter plots for time series EDA?

<details><summary><b>Answer</b></summary>

Yes, scatterplot helps us to visualise the relationship between the variables. For example we can study the relationship between demand and temperature by plotting one series against the other. 

</details>

---

Q. What is the difference between correlation and autocorrelation?

<details><summary><b>Answer</b></summary>

Correlation measures the extent of a linear relationship between two variables, autocorrelation measures the linear relationship between lagged values of a time series.

</details>

---

Q. What is the autocorrelation function (ACF)?

<details><summary><b>Answer</b></summary>

The ACF is a plot of autocorrelation between a variable and itself separated by specified lags. 

<table align='center'>
  <tr>
    <td align="center">
      <img src="img/acf_plot.png" alt= "ACF Plot" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> Autocorrelation Function </td>
  </tr>
</table>

</details>

---

Q. Write the expression for autocorrelation?

<details><summary><b>Answer</b></summary>

$$
r_k = \frac{\sum_{t=k+1}^{T}(y_t - \bar{y})(y_{t-k} - \bar{y})}{\sum_{t=1}^{T}(y_t - \bar{y})^2}
$$

Where $T$ is the length of the time series.

Note that $r_1$ measures the relationship between $y_t$ and $y_{t-1}$, $r_2$ measures the relationship between $y_t$ and $y_{t-2}$ and so on. 

</details>

---

Q. Define following terms:
- Trend
- Seasonal
- Cyclic

<details><summary><b>Answer</b></summary>

*Trend*

A trend exists when there is a long-term increase or decrease in the data. It does not have to be linear.

*Seasonal*

A seasonal pattern occurs when a time series is affected by seasonal factors such as the time of the year or the day of the week. Seasonality is always of a fixed and known period

*Cyclic*

A cycle occurs when the data exhibit rises and falls that are not of a fixed frequency. These fluctuations are usually due to economic conditions, and are often related to the "business cycle". 

</details>

---

Q. How can we check for trend in time series data using ACF plots?

<details><summary><b>Answer</b></summary>

When data have a trend, the autocorrelations for small lags tend to be large and positive because observations nearby in time are also nearby in value. So the ACF of a trended time series tends to have positive values that slowly decrease as the lags increase.

<table align='center'>
  <tr>
    <td align="center">
      <img src="img/acf_trend.png" alt= "ACF Plot For Trend Data" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> ACF plot for data with trend </td>
  </tr>
</table>

</details>

---

Q. How can we check for seasonality in time series data using ACF plots?

<details><summary><b>Answer</b></summary>

When data are seasonal, the autocorrelations will be larger for the seasonal lags (at multiples of the seasonal period) than for other lags.

<table align='center'>
  <tr>
    <td align="center">
      <img src="img/acf_seasonal.png" alt= "ACF Plot For Seasonal Data" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> ACF plot for data with seasonal </td>
  </tr>
</table>

</details>

---

Q. How does the ACF plot looks like if data has both trend and seasonality?

<details><summary><b>Answer</b></summary>

The slow decrease in the ACF as the lags increase is due to the trend, while the “scalloped” shape is due to the seasonality.

<table align='center'>
  <tr>
    <td align="center">
      <img src="img/acf_trend_seasonal.png" alt= "ACF Plot For Seasonal and Trend Data" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> ACF plot for data with seasonal and trend </td>
  </tr>
</table>

</details>

---

Q. What does white noise mean in time series?

<details><summary><b>Answer</b></summary>

Time series that show no autocorrelation are called white noise. 

<table align='center'>
  <tr>
    <td align="center">
      <img src="img/white_noise.png" alt= "White Noise" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> White Noise </td>
  </tr>
</table>

</details>

---

Q. What are the statistical properties of white noise?

<details><summary><b>Answer</b></summary>

- For white noise series, autocorrelation to be close to zero.
- For a white noise series, we expect $95%$ of the spikes in the ACF to lie within $±1.96/\sqrt{T}$ where $T$ is the length of the time series.

</details>

---

Q. How could you check if the given time series is white noise?

<details><summary><b>Answer</b></summary>

We can use the fact that for a white noise series, we expect $95%$ of the spikes in the ACF to lie within $±1.96/\sqrt{T}$ where $T$ is the length of the time series.

It is common to plot these bounds on a graph of the ACF (the blue dashed lines above). If one or more large spikes are outside these bounds, or if substantially more than 5% of spikes are outside these bounds, then the series is probably not white noise.

</details>

---

## Time Series Decomposition

Q. What is time series decomposition?

<details><summary><b>Answer</b></summary>

Splitting a time series into several components, each representing an underlying pattern category.
- A trend-cycle component ($T_t$)
- A seasonal component ($S_t$)
- A residual component ($R_t$)

</details>

---

Q. Can a given time series posses more than one seasonal component?

<details><summary><b>Answer</b></summary>

For some time series (e.g., those that are observed at least daily), there can be more than one seasonal component, corresponding to the different seasonal periods.

</details>

---

Q. What are the benefits of time series decomposition?

<details><summary><b>Answer</b></summary>

- It helps improve understanding of the time series
- It can also be used to improve forecast accuracy.

</details>

---

Q. What kind of adjustments we can do with time series data to simplify the patterns in it?

<details><summary><b>Answer</b></summary>

We can do four kinds of adjustments:
- calendar adjustments
- population adjustments
- inflation adjustments 
- mathematical transformations.

</details>

---

Q. Why is it recommended to make adjustments or transformations to time series data before decomposing it?

<details><summary><b>Answer</b></summary>

The purpose of adjustments and transformations is to simplify the patterns in the historical data by removing known sources of variation, or by making the pattern more consistent across the whole data set. Simpler patterns are usually easier to model and lead to more accurate forecasts.

</details>

---

Q. Why is it recommended to make adjustments or transformations to time series data before decomposing it?

<details><summary><b>Answer</b></summary>

The purpose of adjustments and transformations is to simplify the patterns in the historical data by removing known sources of variation, or by making the pattern more consistent across the whole data set. Simpler patterns are usually easier to model and lead to more accurate forecasts.

</details>

---

Q. What are some common mathematical transformations that can be applied to time series data?

<details><summary><b>Answer</b></summary>

We can apply following transformations to time series depending on the scenarios
- Logarithmic transformations
- Power transformations
- Box-cox transformations

</details>

---

Q. What are the benefits of using mathematical transformations?

<details><summary><b>Answer</b></summary>

Mathematical transformations are techniques used to modify data in ways that make it more suitable for analysis. They can help stabilize variance, reduce skewness, and make relationships within the data more linear or normally distributed.

</details>

---

Q. What is log transformation? 

<details><summary><b>Answer</b></summary>

If we denote the original observations as $y_1,...,y_T$ and the transformed observations as $w_1,w_2,...,w_T$, then $w_t = \log{y_t}$.

</details>

---

Q. In which scenarios we should use log transformations? 

<details><summary><b>Answer</b></summary>

Log transformations reduce right-skewness and stabilizes variance, especially in cases where data values are growing exponentially. It is commonly used when data ranges over several orders of magnitude.

</details>

---

Q. What is power transformations?

<details><summary><b>Answer</b></summary>

It uses mapping as:

$$
y' = y^p
$$

It Increases or decreases the rate of change for different data values. Depending on the power \( p \) (for example, \( p = 2 \) for a square transformation or \( p = -1 \) for a reciprocal transformation), this transformation can reduce skewness or stabilize variance.

</details>

---

Q. What are Box-Cox transformations?

<details><summary><b>Answer</b></summary>

A useful family of transformations, that includes both logarithms and power transformations, is the family of Box-Cox transformations, which depend on the parameter $λ$ and are defined as follows:

$$
y(\lambda) = 
\begin{cases} 
\frac{y^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
\ln(y) & \text{if } \lambda = 0 
\end{cases}
$$

where:
- \( y \) is the original data,
- \( \lambda \) is the transformation parameter.

For \( \lambda = 0 \), the natural logarithm of \( y \) is used instead, which is a common transformation when data has exponential growth or multiplicative seasonality. The value of \( \lambda \) can be estimated to best transform the data for further analysis.

</details>

---

Q. What are additive and multiplicative models in time series decomposition?

<details><summary><b>Answer</b></summary>

In time series decomposition, additive and multiplicative models are used to break down a time series into its components: trend, seasonality, and residual (or noise). 

</details>

---

Q. Explain additive model in time series decomposition?

<details><summary><b>Answer</b></summary>

In an additive model, the time series value at any given time \( Y_t \) is the sum of three components:

- Trend (\( T_t \)) – the long-term increase or decrease in the data.
- Seasonal (\( S_t \)) – the repeating short-term cycle in the data (such as monthly or yearly seasonality).
- Residual (\( R_t \)) – the remaining random variation or noise.

$$
Y_t = T_t + S_t + R_t
$$

</details>

---

Q. When should we use additive model for time series decomposition?

<details><summary><b>Answer</b></summary>

An additive model is appropriate when the trend and seasonal variations are relatively constant over time. This means that the amplitude (size) of seasonal variations does not change with the level of the time series. For instance, if sales increase by a constant amount each year, an additive model might be suitable.

</details>

---

Q. Explain multiplicative model in time series decomposition?

<details><summary><b>Answer</b></summary>

In an multiplicative model, the time series value at any given time \( Y_t \) is the sum of three components:

- Trend (\( T_t \)) – the long-term increase or decrease in the data.
- Seasonal (\( S_t \)) – the repeating short-term cycle in the data (such as monthly or yearly seasonality).
- Residual (\( R_t \)) – the remaining random variation or noise.

$$
Y_t = T_t \times S_t \times R_t 
$$

</details>

---

Q. When should we use multiplicative model for time series decomposition?

<details><summary><b>Answer</b></summary>

A multiplicative model is appropriate when the seasonal variations change proportionally to the trend level. In this case, seasonal fluctuations grow or shrink as the trend rises or falls. For example, if sales increase by a certain percentage each year, a multiplicative model would be more suitable.

</details>

---

Q. How do we determine whether to use an additive or multiplicative model?

<details><summary><b>Answer</b></summary>

- Additive Model: Use when seasonality is roughly constant, regardless of the level of the trend.
- Multiplicative Model: Use when seasonality varies proportionally with the trend.

</details>

---

Q. How does a log transformation allow additive decomposition to approximate a multiplicative decomposition?

<details><summary><b>Answer</b></summary>

An alternative to using a multiplicative decomposition is to first transform the data until the variation in the series appears to be stable over time, then use an additive decomposition. When a log transformation has been used, this is equivalent to using a multiplicative decomposition on the original data because 

$$
y_t = S_t \times T_t \times R_t
$$

On taking $\log$ both side:

$$
\log{y_t} = \log{S_t} + \log{T_t} + \log{R_t}
$$

</details>

---

Q. What is seasonally adjusted data?

<details><summary><b>Answer</b></summary>

If the seasonal component is removed from the original data, the resulting values are the "seasonally adjusted" data. For an additive decomposition, the seasonally adjusted data are given by $y_t - S_t$ and for multiplicative data, the seasonally adjusted values are obtained using $\frac{y_t}{S_t}$

</details>

---

Q. Explain moving average smoothing in time series decomposition?

<details><summary><b>Answer</b></summary>

A moving average of order $m$ can be written as

$$
\hat{T}_{t} = \frac{1}{m}\sum_{j=-k}{k}y_{t+j}
$$

where $m = 2k+1$. That is, the estimate of the trend-cycle at time $t$ is obtained by averaging values of the time series within $k$ periods of $t$. Observations that are nearby in time are also likely to be close in value. Therefore, the average eliminates some of the randomness in the data, leaving a smooth trend-cycle component. 

</details>

---

Q. How does order $m$ of moving average impact the modelling?

<details><summary><b>Answer</b></summary>

The order of the moving average determines the smoothness of the trend-cycle estimate. 

</details>

---

Q. In an m-order moving average, is symmetry important?

<details><summary><b>Answer</b></summary>

In an m-order moving average, symmetry is important because it ensures that each data point is treated equally, minimizing lag and providing a more accurate representation of the trend at a given time. 

</details>

---

Q. Explain weighted moving averages?

<details><summary><b>Answer</b></summary>

Combinations of moving averages result in weighted moving averages. In general, a weighted m-MA can be written as:

$$
\hat{T}_t = \sum^{k}_{j=-k}a_j y_{t+j}
$$

where $k = (m-1)/2$, and weights are given by $[a_{-k},..,a_k]$. It is important that the weights all sum to one and that they are symmetric so that $a_j = a_{-j}$.

</details>

---

Q. What is the major advantage of using weighted moving averages over m-MA?

<details><summary><b>Answer</b></summary>

A major advantage of weighted moving averages is that they yield a smoother estimate of the trend-cycle. Instead of observations entering and leaving the calculation at full weight, their weights slowly increase and then slowly decrease, resulting in a smoother curve.

</details>

---

Q. How can we use m-MA for time series decomposition?

<details><summary><b>Answer</b></summary>

An m-Moving Average (m-MA) can be used for time series decomposition by helping to separate the trend and seasonal components from the data.

1. Identify the Seasonal Period: 
- Determine the period of seasonality in your data, such as daily, monthly, or quarterly, depending on the time series. The chosen value of $m$ is usually equal to this seasonal period.
2. Calculate the Moving Average: 
- Apply an m-point moving average to smooth the data. The choice of m depends on the frequency of seasonality. For example, with monthly data and yearly seasonality, you’d use a 12-point moving average.
- For symmetric smoothing, use a centered moving average, where you calculate the average over equal numbers of points before and after a central point
3. Extract the Trend Component:
- The resulting moving average values represent the trend component, which shows the general direction of the series (upward, downward, or flat).
4. Isolate the Seasonal Component:
- To find the seasonal component, divide the original time series values by the trend (for a multiplicative model) or subtract the trend values (for an additive model).
- This can be done across several periods to get the average seasonal pattern, which smooths out random variations.
5. Calculate the Residual Component:
- After extracting the trend and seasonal components, the residual (or irregular) component can be determined by subtracting the seasonal component from the detrended data in the additive model, or dividing it in the multiplicative model.
- The residual component represents the noise or random variation left in the data after removing the trend and seasonality.

</details>

---

Q. What are the limitations of classical time series decomposition?

<details><summary><b>Answer</b></summary>

Classical time series decomposition has these main limitations:

1. Missing Trend Estimates: It doesn’t estimate the trend-cycle for the first and last few data points.
2. Over-Smoothing: Rapid changes in data are often smoothed out, losing detail.
3. Fixed Seasonality: Assumes seasonality is constant over time, which fails with evolving patterns.
4. Outlier Sensitivity: Not robust to unusual or extreme values, which can skew the results.

</details>

---

Q. How does STL decomposition work?

<details><summary><b>Answer</b></summary>

STL is a versatile and robust method for decomposing time series. STL is an acronym for "Seasonal and Trend decomposition using Loess", while loess is a method for estimating nonlinear relationships. STL was designed to handle data that exhibit non-linear patterns and allows for changing seasonality, unlike classical decomposition methods.

Here's how STL decomposition works:

1. Loess (Locally Estimated Scatterplot Smoothing): This is a non-parametric technique that uses local weighted regression to smooth parts of the data. STL uses Loess to estimate both the trend and seasonal components.
2. Seasonal Estimation: 
- STL first removes the rough trend by applying a Loess smoother to the entire series. This detrended series is then used to estimate the seasonal component, again using Loess smoothing but focusing on the seasonal cycle's length.
- The seasonality is allowed to change over time, and STL handles this by breaking the series into cycles and smoothing each separately.
3. Trend Estimation:
- Once the seasonal component is subtracted from the original data, what remains (original minus seasonal) is used to estimate the trend using another Loess smoother.
- This step focuses on longer periods than the seasonal estimation to capture the overall direction or trend without short-term fluctuations.
4. Residual Calculation:
- The residual component is simply calculated by subtracting both the estimated seasonal and trend components from the original time series.
5. Iterative Procedure:
- STL performs these steps iteratively, refining the estimates of trend and seasonality to minimize the residual component. This iterative approach allows STL to adapt to complex and changing patterns in the data.

</details>

---

Q. What are the advantages of using STL over classical decomposition?

<details><summary><b>Answer</b></summary>

STL has several advantages over classical decomposition:
- The seasonal component is allowed to change over time, and the rate of change can be controlled by the user.
- The smoothness of the trend-cycle can also be controlled by the user.
- It can be robust to outliers (i.e., the user can specify a robust decomposition), so that occasional unusual observations will not affect the estimates of the trend-cycle and seasonal components.

</details>

---

Q. What are the limitations of STL decomposition?

<details><summary><b>Answer</b></summary>

STL (Seasonal-Trend Decomposition using Loess) is a powerful decomposition method, but it does have some limitations:
- Computational Intensity: It requires significant computational resources, especially for large datasets.
- Lack of Forecasting Capability: STL doesn't directly provide forecasting models; it's primarily for decomposition.
- It does not handle trading day or calendar variation automatically, and it only provides facilities for additive decompositions.

</details>

---

Q. How can we use time series decomposition to  measure the strength of trend and seasonality in a time series?

<details><summary><b>Answer</b></summary>

A time series decomposition can be used to measure the strength of trend and seasonality in a time series:

$$
y_t = T_t + S_t + R_t 
$$

*Strength of Trend*

For strongly trended data, the seasonally adjusted data should have much more variation than the remainder component.

$$
F_T = max(0, 1 - \frac{\text{Var}(R_t)}{\text{Var}(T_t + R_t)})
$$

This will give a measure of the strength of the trend between 0 and 1.

*Strength of seasonality*

The strength of seasonality is defined similarly, but with respect to the detrended data rather than the seasonally adjusted data: 

$$
F_S = max(0, 1 - \frac{\text{Var}(R_t)}{\text{Var}(S_t + R_t)})
$$

A series with seasonal strength $F_S$ close to 0 exhibits almost no seasonality.


</details>

---

## Benchmark Forecasting Methods

Q. What are some simple forecasting methods?

<details><summary><b>Answer</b></summary>

- Mean method
- Naive Method
- Seasonal Naive Method
- Drift Method

</details>

---

Q. Explain mean method in time series forecasting?

<details><summary><b>Answer</b></summary>

Mean method assumes that forecasts of all future values are equal to average of historical data. If we let the historical data denoted by $y_1,..,y_T$, then we can write forecasts as

$$
\hat{y}_{T+h|T} = \bar{y} = (y_1 + ... + y_T)/T
$$

<table align='center'>
  <tr>
    <td align="center">
      <img src="img/mean_method.png" alt= "Mean method forecasts applied to clay brick production in Australia." style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> Mean method forecasts applied to clay brick production in Australia. </td>
  </tr>
</table>

</details>

---

Q. How does naive method works in forecasting?

<details><summary><b>Answer</b></summary>

For naïve forecasts, we simply set all forecasts to be the value of the last observation. That is,

$$
\hat{y}_{T+h|T} = y_T 
$$

Note that naive method is also called random walk forecasts.

<table align='center'>
  <tr>
    <td align="center">
      <img src="img/naive_method.png" alt= "Naïve forecasts applied to clay brick production in Australia." style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> Naïve forecasts applied to clay brick production in Australia. </td>
  </tr>
</table>

</details>

---

Q. How does seasonal naive method works in forecasting?

<details><summary><b>Answer</b></summary>

We  set each forecast to be equal to the last observed value from the same season (e.g., the same month of the previous year). 

$$
\hat{y}_{T+h|T} = y_{T+h - m(k+1)} 
$$

Where m = the seasonal period, and $k$ is the integer part of $(h-1)/m$ (i.e., the number of complete years in the forecast period prior to time $T+h$). For example, with monthly data, the forecast for all future February values is equal to the last observed February value.

<table align='center'>
  <tr>
    <td align="center">
      <img src="img/seasonal_naive_method.png" alt= "Seasonal naïve forecasts applied to clay brick production in Australia." style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> Seasonal naïve forecasts applied to clay brick production in Australia. </td>
  </tr>
</table>

</details>

---

Q. How does drift method works in forecasting?

<details><summary><b>Answer</b></summary>

A variation on the naïve method is to allow the forecasts to increase or decrease over time, where the amount of change over time (called the drift) is set to be the average change seen in the historical data. 

Forecast for time $T+h$ is given by

$$
\hat{y}_{T+h|T} = y_T + \frac{h}{T-1} \sum_{t=2}^{T}(y_t - y_{t-1}) = Y_T + h(\frac{y_t - y_1}{T - 1})
$$

Where m = the seasonal period, and $k$ is the integer part of $(h-1)/m$ (i.e., the number of complete years in the forecast period prior to time $T+h$). For example, with monthly data, the forecast for all future February values is equal to the last observed February value.


<table align='center'>
  <tr>
    <td align="center">
      <img src="img/drift_method.png" alt= "Drift forecasts applied to clay brick production in Australia." style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> Drift forecasts applied to clay brick production in Australia. </td>
  </tr>
</table>

</details>

---

Q. What do you mean by residual in time series model?

<details><summary><b>Answer</b></summary>

The "residuals" in a time series model are what is left over after fitting a model. The residuals are equal to the difference between the observations and the corresponding fitted values:

$$
e_t = y_t - \hat{y}_t
$$

Residuals are useful in checking whether a model has adequately captured the information in the data. 

</details>

---

Q. What properties should innovation residuals have to indicate a good forecasting method?

<details><summary><b>Answer</b></summary>

A good forecasting method will yield innovation residuals with the following properties:

Essential properties:
- The innovation residuals are uncorrelated. If there are correlations between innovation residuals, then there is information left in the residuals which should be used in computing forecasts.
- The innovation residuals have zero mean. If they have a mean other than zero, then the forecasts are biased.
Good to have:
- The innovation residuals have constant variance. This is known as “homoscedasticity”.
- The innovation residuals are normally distributed.

</details>

---

Q. What properties should innovation residuals have to indicate a good forecasting method?

<details><summary><b>Answer</b></summary>

A good forecasting method will yield innovation residuals with the following properties:

Essential properties:
- The innovation residuals are uncorrelated. If there are correlations between innovation residuals, then there is information left in the residuals which should be used in computing forecasts.
- The innovation residuals have zero mean. If they have a mean other than zero, then the forecasts are biased.
Good to have:
- The innovation residuals have constant variance. This is known as “homoscedasticity”.
- The innovation residuals are normally distributed.

</details>

---

Q. How do you determine the prediction interval for forecasted values?

<details><summary><b>Answer</b></summary>

Most time series models produce normally distributed forecasts — that is, we assume that the distribution of possible future values follows a normal distribution. A prediction interval gives an interval within which we expect $y_t$ to lie with a specified probability. For example, assuming that distribution of future observations is normal, a $95\%$  prediction interval for the h-step forecast is:

$$
\hat{y}_{T+h|T} \pm 1.96\hat{\sigma}_h
$$

where $\hat{\sigma}_h$ is an estimate of the standard deviation of the  h-step forecast distribution.

More generally, a prediction interval can be written as

$$
\hat{y}_{T+h|T} \pm c\hat{\sigma}_h
$$

where the multiplier $c$ depends on the coverage probability.

</details>

---

Q. How do you determine the prediction interval for forecasted values?

<details><summary><b>Answer</b></summary>

Most time series models produce normally distributed forecasts — that is, we assume that the distribution of possible future values follows a normal distribution. A prediction interval gives an interval within which we expect $y_t$ to lie with a specified probability. For example, assuming that distribution of future observations is normal, a $95\%$  prediction interval for the h-step forecast is:

$$
\hat{y}_{T+h|T} \pm 1.96\hat{\sigma}_h
$$

where $\hat{\sigma}_h$ is an estimate of the standard deviation of the  h-step forecast distribution.

More generally, a prediction interval can be written as

$$
\hat{y}_{T+h|T} \pm c\hat{\sigma}_h
$$

where the multiplier $c$ depends on the coverage probability.

</details>

---

Q. Express the standard deviation of the forecast distribution in case of one step prediction?

<details><summary><b>Answer</b></summary>

When forecasting one step ahead, the standard deviation of the forecast distribution can be estimated using the standard deviation of the residuals given by 

$$
\hat{\sigma} = \sqrt{\frac{1}{T - K - M}\sum_{t=1}^{T}e_{t}^2}
$$

where $K$ is the number of parameters estimated in the forecasting method, and $M$ is the number of missing values in the residuals.

</details>

---

Q. What happens with prediction intervals in case of multi-step forecasting?

<details><summary><b>Answer</b></summary>

Prediction intervals usually increase in length as the forecast horizon increases. The further ahead we forecast, the more uncertainty is associated with the forecast, and thus the wider the prediction intervals.

</details>

---

Q. For benchmark methods write the standard deviation expression for h-step forecast distribution?

<details><summary><b>Answer</b></summary>

For the four benchmark methods, it is possible to mathematically derive the forecast standard deviation under the assumption of uncorrelated residuals.

$$
\begin{table}[H]
\centering
\begin{tabular}{|c|c|}
\hline
\textbf{Forecasting Method} & \textbf{h-step forecast standard deviation} \\ \hline
Mean                        & $\hat{\sigma}_h = \hat{\sigma} \sqrt{1 + \frac{1}{T}}$                 \\ \hline
Naïve                       & $\hat{\sigma}_h = \hat{\sigma} \sqrt{h}$                               \\ \hline
Seasonal Naïve              & $\hat{\sigma}_h = \hat{\sigma} \sqrt{k + 1}$                           \\ \hline
Drift                       & $\hat{\sigma}_h = \hat{\sigma} \sqrt{h \left(1 + \frac{h}{T - 1}\right)}$ \\ \hline
\end{tabular}
\caption{Standard Deviation Equations for Different Forecasting Methods}
\label{tab:forecast_sd}
\end{table}
$$

</details>

---

Q. If the residuals from a fitted forecasting model do not exhibit a normal distribution, how would you establish prediction intervals for the forecasted values? 

<details><summary><b>Answer</b></summary>

When assuming a normal distribution for residuals is not suitable, an alternative approach is to use bootstrapping. This method assumes that the residuals are uncorrelated and have a consistent variance.

</details>

---

Q. Can time series decomposition be utilized for forecasting, and if so, what is the method for doing so?

<details><summary><b>Answer</b></summary>

Yes, Time series decomposition can be a useful step in producing forecasts. Assuming an additive decomposition, the decomposed time series can be written as

$$
y_t = \hat{S}_t + \hat{A}_t
$$

Where $\hat{A}_t = \hat{T}_t + \hat{R}_t$ is the seasonally adjusted component. 

To forecast a decomposed time series, we forecast the seasonal component, $\hat{S}_t$, and the seasonally adjusted component $\hat{A}_t$, separately. It is usually assumed that the seasonal component is unchanging, or changing extremely slowly, so it is forecast by simply taking the last year of the estimated component. In other words, a seasonal naïve method is used for the seasonal component.

To forecast the seasonally adjusted component, any non-seasonal forecasting method may be used. For example, the drift method, or Holt’s method, or a non-seasonal ARIMA model may be used.

</details>

---

Q. How does forecast errors differ from residuals?

<details><summary><b>Answer</b></summary>

Forecast errors are different from residuals in two ways. First, residuals are calculated on the training set while forecast errors are calculated on the test set. Second, residuals are based on one-step forecasts while forecast errors can involve multi-step forecasts.

</details>

---

Q. What are different techniques for measuring forecast accuracy?

<details><summary><b>Answer</b></summary>

We can measure forecast accuracy by summarising the forecast errors in following ways:

*Scale-dependent errors*
- Mean absolute error(MAE)
- Root mean squared error(RMSE)

*Percentage errors*
- Mean absolute percentage error(MAPE)

*Scaled Errors*

Scaled errors were proposed by Hyndman & Koehler (2006) as an alternative to using percentage errors when comparing forecast accuracy across series with different units. They proposed scaling the errors based on the training MAE from a simple forecast method.


</details>

---

Q. Is it feasible to apply the cross-validation technique to evaluate the accuracy of forecasts?

<details><summary><b>Answer</b></summary>

Yes, In this procedure, there are a series of test sets, each consisting of a single observation. The corresponding training set consists only of observations that occurred prior to the observation that forms the test set.

<table align='center'>
  <tr>
    <td align="center">
      <img src="img/time_series_cv.png" alt= "Cross-validation for time series forecasts" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> Cross-validation for time series forecasts </td>
  </tr>
</table>

With time series forecasting, one-step forecasts may not be as relevant as multi-step forecasts. In this case, the cross-validation procedure based on a rolling forecasting origin can be modified to allow multi-step errors to be used. Suppose that we are interested in models that produce good 4-step-ahead forecasts. Then the corresponding diagram is shown below:

<table align='center'>
  <tr>
    <td align="center">
      <img src="img/time_series_cv_2.png" alt= "Cross-validation for time series multi-step forecasts" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> Cross-validation for time series multi-step forecasts </td>
  </tr>
</table>

</details>

---

## Time Series Regression Models

Q. What assumptions do we make when using a linear regression model for forecasting?

<details><summary><b>Answer</b></summary>

When using a linear regression model, we assume:

1. Model Approximation: The model is a reasonable approximation to reality. This implies that the relationship between the forecast variable and predictor variables is linear.

2. Assumptions about the Errors:
   - Mean Zero: The errors have a mean of zero to avoid systematic bias in forecasts.
   - No Autocorrelation: The errors are not autocorrelated, ensuring forecasts are efficient without missed information in the data.
   - Unrelated to Predictors: The errors are unrelated to predictor variables, suggesting that all relevant information is captured within the model's systematic part.
   - Normal Distribution with Constant Variance: It is helpful for the errors to be normally distributed with constant variance (\( \sigma^2 \)) to facilitate prediction interval calculations.

</details>

---

Q. Explain least squares principle?

<details><summary><b>Answer</b></summary>

The least squares principle provides a way of choosing the coefficients effectively by minimising the sum of the squared errors. That is, we choose the values of $\beta_0,\beta_1,..,\beta_k$ that minimise 

$$
\sum_{t=1}^T \eta_{t}^2 = \sum_{t=1}^T(y_t - \beta_0 - \beta_1x_{1, t} - ... - \beta_k x_{k, t})^2
$$

This is called least squares estimation because it gives the least value for the sum of squared errors. 

</details>

---

Q. What are some typical predictors used in time series regression models?

<details><summary><b>Answer</b></summary>

There are several useful predictors that occur frequently when using regression for time series data.
- Trend: It is common for time series data to be trending. A linear trend can be modelled by simply using $x_{1, t}=t$ as predictor,

$$
y_t = \beta_0 + \beta_1t + \eta_t
$$
- Dummy variables
    - Public holiday 
- Seasonal dummy variables
    - Day of the week
    - Weekends
    - Week of the month
    - Month
    - Quarter
- Intervention variables: It is often necessary to model interventions that may have affected the variable to be forecast.
    - Competitor activity
    - Advertising expenditure
    - Industrial action
- Trading days: The number of trading days in a month can vary considerably and can have a substantial effect on sales data.
    - number of Mondays/Sundays in month
- Distributed lags
- Rolling stats

</details>

---

Q. What is Akaike's Information Criterion (AIC)?

<details><summary><b>Answer</b></summary>

AIC is defined as:

$$
\text{AIC} = T\log(\frac{SSE}{T}) + 2(k+2)
$$

$$
\text{SSE} = \sum_{t=1}^T \eta_{t}^2
$$

where $T$ is the number of observations used for estimation and $k$ is the number of predictors in the model.

The $k+2$ part of the equation occurs because there are $k+2$ parameters in the model: the $k$ coefficients for the predictors, the intercept and the variance of the residuals. 

</details>

---

Q. How can the AIC score be interpreted?

<details><summary><b>Answer</b></summary>

The model with the minimum value of the AIC is often the best model for forecasting. 

</details>

---

Q. Why do we need to adjust bias in AIC score?

<details><summary><b>Answer</b></summary>
For small values of $T$, the AIC tends to select too many predictors, and so a bias-corrected version of the AIC has been developed.

$$
\text{AIC}_c = \text{AIC} + \frac{2(k+2)(k+3)}{(T-k-3)}
$$

</details>

---

Q. What is Bayesian Information Criterion (BIC)?

<details><summary><b>Answer</b></summary>

Schwarz’s Bayesian Information Criterion (usually abbreviated to BIC, SBIC or SC):

$$
\text{BIC} = T\log{\frac{SSE}{T}} + (k+2)\log(T)
$$

</details>

---

Q. How does AIC differs from BIC?

<details><summary><b>Answer</b></summary>

BIC penalizes the number of parameters more heavily than the AIC. Although  If the value of $T$ is large enough, both will lead to the same model.

</details>

---




## Exponential Smoothing

## ARIMA Models

## Prophet Model

## Vector Autoregressions

## Neural Networks