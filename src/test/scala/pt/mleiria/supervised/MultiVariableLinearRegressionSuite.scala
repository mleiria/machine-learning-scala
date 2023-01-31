package pt.mleiria.supervised

import breeze.linalg.{DenseMatrix, DenseVector}
import pt.mleiria.preprocess.Scaler
import pt.mleiria.utils.{IOUtils, MathUtils, MatrixUtils}
import spire.implicits.eqOps

class MultiVariableLinearRegressionSuite extends munit.FunSuite:

  test("gradientDescent") {
    // Dados de treino do nosso modelo
    val xTrain: DenseMatrix[Double] = DenseMatrix((2104.0, 5.0, 1.0, 45.0), (1416.0, 3.0, 2.0, 40.0), (852.0, 2.0, 1.0, 35.0))
    val yTrain: DenseVector[Double] = DenseVector(460.0, 232.0, 178.0)

    val m: Int = xTrain.rows
    val n: Int = xTrain.cols
    // Initialize parameters
    val wIn: DenseVector[Double] = DenseVector.zeros[Double](n)
    val bIn = 0.0
    // some gradient descent settings
    val iterations = 1000
    val alpha = 5.0e-7
    // Run gradient descent
    val gd: MultiVariableLinearRegression = MultiVariableLinearRegression(xTrain, yTrain)

    val (wFinal, bFinal, jHistory) = gd.fit(bIn, alpha, iterations, wIn)

    val bFinalRounded = MathUtils.roundDoubleScalar(bFinal)
    val wFinalRounded = MathUtils.roundDoubleVector(wFinal)

    assert(bFinalRounded == 0.0)

    assert(wFinalRounded(0) == 0.2)
    assert(wFinalRounded(1) == 0.0)
    assert(wFinalRounded(2) == -0.01)
    assert(wFinalRounded(3) == -0.07)

  }

  test("gradientDescentWithScaler") {
    // Load the dataset from csv file
    val xy: DenseMatrix[Double] = IOUtils.readCsvFromResourcesToMatrix("data/houses.txt")
    // Split the data into xTrain and yTrain
    val (xTrain, yTrain) = MatrixUtils.loadXYDatasetFromMatrix(xy)
    val (xTrainNorm, mean, stdev) = Scaler.zScoreNormalizer(xTrain)
    val iterations = 1000
    val alpha = 1.0e-1
    val bIn = 0.0
    val gd = MultiVariableLinearRegression.apply(xTrainNorm, yTrain)
    val (wNorm, bNorm, history) = gd.fit(bIn, alpha, iterations)

    assert(MathUtils.roundDoubleScalar(wNorm(0)) == 111.17)
    assert(MathUtils.roundDoubleScalar(wNorm(1)) == -21.58)
    assert(MathUtils.roundDoubleScalar(wNorm(2)) == -32.83)
    assert(MathUtils.roundDoubleScalar(wNorm(3)) == -37.97)
    assert(MathUtils.roundDoubleScalar(bNorm) == 362.24)
  }



