
import org.apache.spark.SparkConf

import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.feature.IDF
import org.apache.spark.rdd.RDD
import java.text.Normalizer.Form
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.sql.SQLContext
object Mlib {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setMaster("local[*]").setAppName("Mlib")
    val sc = new SparkContext(conf)
    val spam = sc.textFile("files/spam.txt")
    val ham = sc.textFile("files/ham.txt")
    val tf = new HashingTF(numFeatures = 100)

    val spamFeatures = spam.map(email => tf.transform(email.split(" ")))
    val hamFeatures = ham.map(email => tf.transform(email.split(" ")))

    val positiveExamples = spamFeatures.map(features => LabeledPoint(0, features))
    val negativeExamples = hamFeatures.map(features => LabeledPoint(1, features))
    val trainingData = positiveExamples ++ negativeExamples

    val model = NaiveBayes.train(trainingData, lambda = 1.0, modelType = "multinomial")

    // Dear Spark Learner, Thanks so much for attending the Spark Summit 2014 Dear !
    val positiveExample = tf.transform(" Dear Spark Learner, Thanks so much for attending the Spark Summit 2014 Dear ! ...".split(" "))
    val negativeExample = tf.transform(" Here we can deliver you a bomb. hatred should be rejected ! ...".split(" "))

    val status = if (model.predict(positiveExample) == 0.0) "This is Spam" else "This Email is Good";

    //or 
    // val status = if (model.predict(negativeExample) == 0.0) "This is Spam" else "This Email is Good";

    
    print(status)
 
  }
}