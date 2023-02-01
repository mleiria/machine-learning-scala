package pt.mleiria.supervised

import javax.print.attribute.standard.JobHoldUntil
import scala.annotation.tailrec

/**
 *
 * @param x Training features
 * @param y Target data
 */
class UnivariateLinearRegression(val x: Array[Double], val y: Array[Double]) {

  // Number of training examples
  val m: Int = x.length

  /**
   *
   * @param w Model parameters
   * @param b Model parameters
   * @return total cost (Double): The cost of using w,b as the parameters for linear regression
   *         to fit the data points in x and y
   */
  def computeCost(w: Double, b: Double): Double = {

    @tailrec
    def loop(costSum: Double, i: Int): Double = {
      if i == m then {
        val totalCost = (1.0 / (2.0 * m)) * costSum
        totalCost
      } else {
        val fwb = w * x(i) + b
        val cost = math.pow(fwb - y(i), 2)
        loop(costSum + cost, i + 1)
      }
    }

    loop(0, 0)
  }

  /**
   * Computes the gradient for linear regression
   *
   * @param w Model parameters
   * @param b Model parameters
   * @return dJdw (Double): The gradient of the cost w.r.t. the parameters w
   *         dJdb (Double): The gradient of the cost w.r.t. the parameter b
   */
  def computeGradient(w: Double, b: Double): (Double, Double) = {

    @tailrec
    def loop(dJdw: Double, dJdb: Double, i: Int): (Double, Double) = {
      if i == m then {
        (dJdw / m, dJdb / m)
      } else {
        val fwb = w * x(i) + b
        val dJdwi = (fwb - y(i)) * x(i)
        val dJdbi = fwb - y(i)
        loop(dJdw + dJdwi, dJdb + dJdbi, i + 1)
      }
    }

    loop(0.0, 0.0, 0)
  }

  /**
   *
   * @param wIn      Initial value for model parameters
   * @param bIn      Initial value for model parameters
   * @param alpha    Learning rate
   * @param numIters Number of iterations to run gradient descent
   * @param gradFunc Gradient Fucntion
   * @param costFunc Cost Function
   * @return w: Updated value of parameter after running gradient descent
   *         b: Updated value of parameter after running gradient descent
   *         JHistory: History of cost values
   */
  def gradientDescent(wIn: Double, bIn: Double, alpha: Double, numIters: Int,
                      gradFunc: (Double, Double) => (Double, Double),
                      costFunc: (Double, Double) => Double):
  (Double, Double, Array[Double]) = {

    // An array to store cost J and w's at each iteration primarily for graphing later
    val jHistory = new Array[Double](numIters)

    @tailrec
    def loop(w: Double, b: Double, counter: Int): (Double, Double, Array[Double]) = {
      if counter == numIters then {
        (w, b, jHistory)
      } else {
        // Calculate the gradient and update the parameters using gradient function
        val res = gradFunc(w, b)
        val dJdw = res._1
        val dJdb = res._2

        // Update parameters
        val wNext = w - alpha * dJdw
        val bNext = b - alpha * dJdb

        // Save cost J at each iteration
        jHistory(counter) = costFunc(wNext, bNext)
        if counter % math.ceil(numIters / 10) == 0 then
          printf("| %-15s | %-30s ", s"Iteration: ${counter}", s"Cost: ${jHistory(counter)}")
          println()
        loop(wNext, bNext, counter + 1)
      }
    }

    loop(wIn, bIn, 0)
  }

  /**
   *
   * @param alpha    Learning rate
   * @param numIters umber of iterations to run gradient descent
   * @return w: Updated value of parameter after running gradient descent
   *         b: Updated value of parameter after running gradient descent
   *         JHistory: History of cost values
   */
  def fit(alpha: Double, numIters: Int): (Double, Double, Array[Double]) = {
    gradientDescent(0.0, 0.0, alpha, numIters, computeGradient, computeCost)
  }

  /**
   *
   * @param wIn      Initial values for model parameters
   * @param bIn      Initial values for model parameters
   * @param alpha    Learning rate
   * @param numIters umber of iterations to run gradient descent
   * @return w: Updated value of parameter after running gradient descent
   *         b: Updated value of parameter after running gradient descent
   *         JHistory: History of cost values
   */
  def fit(wIn: Double, bIn: Double, alpha: Double, numIters: Int): (Double, Double, Array[Double]) = {
    gradientDescent(wIn, bIn, alpha, numIters, computeGradient, computeCost)
  }


}

object UnivariateLinearRegression {

  def apply(x: Array[Double], y: Array[Double]) = {
    new UnivariateLinearRegression(x, y)
  }

}