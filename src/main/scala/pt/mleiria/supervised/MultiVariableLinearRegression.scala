package pt.mleiria.supervised

import breeze.linalg.*

import scala.annotation.tailrec

/**
 *
 * @param x : (m, n): Data: m examples with n features
 * @param y : (n, 1): Target data
 */
class MultiVariableLinearRegression(x: DenseMatrix[Double], y: DenseVector[Double]) {

  // Number of training examples
  val m: Int = x.rows
  // Number of features
  val n: Int = x.cols

  /**
   * Compute cost
   *
   * @param w : (n, 1) Weight vector.Model parameters
   * @param b : Double. Model parameter
   * @return the cost
   */
  def computeCost(w: DenseVector[Double], b: Double): Double = {

    @tailrec
    def loop(costSum: Double, i: Int): Double = {
      if (i == m) {
        val totalCost = (1.0 / (2.0 * m)) * costSum
        totalCost
      } else {
        val fWbi = (x(i, ::) * w) + b
        val cost = math.pow(fWbi - y(i), 2)
        loop(costSum + cost, i + 1)
      }
    }

    loop(0.0, 0)
  }

  /**
   * Computes the gradient for linear regression
   *
   * @param w Model parameters
   * @param b Model parameters
   * @return dJdw (Double): The gradient of the cost w.r.t. the parameters w
   *         dJdb (Double): The gradient of the cost w.r.t. the parameter b
   */
  def computeGradient(w: DenseVector[Double], b: Double): (DenseVector[Double], Double) = {
    val m = x.rows
    val n = x.cols
    val dJdw = DenseVector.zeros[Double](n)

    @tailrec
    def loop(dJdb: Double, i: Int): Double = {
      if (i == m) {
        dJdb
      } else {
        val err = (x(i, ::) * w + b) - y(i)
        for (j <- 0 until n) {
          dJdw(j) = dJdw(j) + err * x(i, j)
        }
        loop(dJdb + err, i + 1)
      }
    }

    val dJdb = loop(0.0, 0)

    (dJdw / m.toDouble, dJdb / m.toDouble)
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
  def gradientDescent(wIn: DenseVector[Double], bIn: Double, alpha: Double, numIters: Int,
                      gradFunc: (DenseVector[Double], Double) => (DenseVector[Double], Double),
                      costFunc: (DenseVector[Double], Double) => Double):
  (DenseVector[Double], Double, Array[Double]) = {

    // An array to store cost J and w's at each iteration primarily for graphing later
    val jHistory = new Array[Double](numIters)

    @tailrec
    def loop(w: DenseVector[Double], b: Double, counter: Int):
    (DenseVector[Double], Double, Array[Double]) = {

      if (counter == numIters) {
        (w, b, jHistory)
      } else {
        // Calculate the gradient and update the parameters
        val (dJdw, dJdb) = computeGradient(w, b)
        // Update Parameters using w, b, alpha and gradient
        val wNext = w - alpha * dJdw
        val bNext = b - alpha * dJdb
        jHistory(counter) = computeCost(w, b)
        // Print cost every at intervals 10 times
        if (counter % math.ceil(counter / 10) == 0) {
          printf("| %-15s | %-30s ", s"Iteration: ${counter}", s"Cost: ${jHistory(counter)}")
          println()
        }
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
  def fit(alpha: Double, numIters: Int): (DenseVector[Double], Double, Array[Double]) = {
    gradientDescent(DenseVector.zeros[Double](n), 0.0, alpha, numIters, computeGradient, computeCost)
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
  def fit(wIn: DenseVector[Double], bIn: Double, alpha: Double, numIters: Int):
  (DenseVector[Double], Double, Array[Double]) = {

    gradientDescent(wIn, bIn, alpha, numIters, computeGradient, computeCost)
  }
}

object MultiVariableLinearRegression {

  def apply(x: DenseMatrix[Double], y: DenseVector[Double]) = {
    new MultiVariableLinearRegression(x, y)
  }

}
