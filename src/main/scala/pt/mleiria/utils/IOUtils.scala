package pt.mleiria.utils

import breeze.linalg.*

import java.io.*
import scala.io.Source

object IOUtils {


  /**
   * Reads a csv file to a DenseMatrix
   *
   * @param csvFile the path and file to load
   * @return DenseMatrix
   */
  def readCsvToMatrix(csvFile: String): DenseMatrix[Double] =
    csvread(new File(csvFile), ',')

  /**
   * Reads a csv file (one column) to a DenseVector
   *
   * @param csvFile the path and file to load
   * @return DenseVector
   */
  def readCsvToVector(csvFile: String): DenseVector[Double] = {
    val numRows: Int = scala.io.Source.fromFile(csvFile).getLines.size
    val arr: Array[Double] = scala.io.Source.fromFile(csvFile).getLines.toArray.map(elem => elem.toDouble)
    DenseVector[Double](arr)
  }

  /**
   * Reads a csv file from resources folder to a DenseMatrix
   *
   * @param csvFile the path inside the resources and file to load
   * @return DenseMatrix
   */
  def readCsvFromResourcesToMatrix(csvFile: String): DenseMatrix[Double] =
    val resource = Source.fromResource(csvFile)
    val lines: Iterator[String] = resource.getLines
    val tmp = lines.map(l => l.split(",").map(str => str.toDouble)).toList
    DenseMatrix(tmp: _*)

}
