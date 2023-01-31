import breeze.linalg.*
import breeze.stats.meanAndVariance

import scala.io.Source


@main def hello: Unit =
  println("hello scala 3")
  testBroadcast()



def testResource():Unit={
  val resource = Source.fromResource("data/houses.txt")
  val lines: Iterator[String] = resource.getLines
  val tmp = lines.map(l => l.split(",").map(str => str.toDouble)).toList
  val m = DenseMatrix(tmp: _*)
  println(m)
}

def testBroadcast():Unit = {
  //(3 x 4)
  val xTrain: DenseMatrix[Double] = DenseMatrix((4.0, 5.0, 6.0, 7.0), (8.0, 9.0, 10.0, 11.0), (12.0, 13.0, 14.0, 15.0))
  println(xTrain)
  val yTrain: DenseVector[Double] = DenseVector(1.0, 2.0, 3.0, 4.0)
  println(yTrain)

  val res = xTrain(*, ::) - yTrain
  println(res)
  //println(res(::, *) / yTrain)
}






