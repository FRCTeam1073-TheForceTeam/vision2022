import edu.wpi.first.vision.VisionPipeline;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableEntry;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Range;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.MatOfPoint;
import org.opencv.imgproc.*;
import org.opencv.imgcodecs.Imgcodecs;
import edu.wpi.cscore.CvSource;

public class IndexerTracker implements VisionPipeline {
    public int frameCounter;
    private NetworkTableInstance nti;
    private NetworkTable indexerTable;
    private NetworkTableEntry total;
    private NetworkTableEntry current;
    private NetworkTableEntry next;
    private NetworkTableEntry redHMin;
    private NetworkTableEntry redHMax;
    private NetworkTableEntry redSMin;
    private NetworkTableEntry redSMax;
    private NetworkTableEntry redVMin;
    private NetworkTableEntry redVMax;
    private NetworkTableEntry blueHMin;
    private NetworkTableEntry blueHMax;
    private NetworkTableEntry blueSMin;
    private NetworkTableEntry blueSMax;
    private NetworkTableEntry blueVMin;
    private NetworkTableEntry blueVMax;
    private CvSource output;
    private Mat hsvImage;
    private Mat blueMask;
    private Mat redMask;
    private Mat outputImage;
    private Mat erosionKernel;
    private Mat nextMat;
    private Mat currentMat;

    public class IndexerData{
      public void clear(){
        x = 0;
        y = 0; 
        width = 0;
        height = 0;
        area = 0;
        isRed = false;
      }
      public double x;
      public double y;
      public double width;
      public double height;
      public double area;
      public boolean isRed;
    }

    //private SimpleBlobDetectorParams blobParam

    public IndexerTracker(NetworkTableInstance ntinst, CvSource output_){
        nti = ntinst;
        indexerTable = nti.getTable("INDEXER");
        total = indexerTable.getEntry("Total Cargo");
        total.setDouble(0);
        current = indexerTable.getEntry("Current Cargo");
        current.setDouble(0);
        next = indexerTable.getEntry("Next Cargo");
        next.setDouble(0);
        redHMin = indexerTable.getEntry("Red H Min");
        redHMin.setDouble(0);
        redHMax = indexerTable.getEntry("Red H Max");
        redHMax.setDouble(30);
        redSMin = indexerTable.getEntry("Red S Min");
        redSMin.setDouble(90);
        redSMax = indexerTable.getEntry("Red S Max");
        redSMax.setDouble(255);
        redVMin = indexerTable.getEntry("Red V Min");
        redVMin.setDouble(60);
        redVMax = indexerTable.getEntry("Red V Max");
        redVMax.setDouble(252);
        blueHMin = indexerTable.getEntry("Blue H Min");
        blueHMin.setDouble(0);
        blueHMax = indexerTable.getEntry("Blue H Max");
        blueHMax.setDouble(30);
        blueSMin = indexerTable.getEntry("Blue S Min");
        blueSMin.setDouble(90);
        blueSMax = indexerTable.getEntry("Blue S Max");
        blueSMax.setDouble(255);
        blueVMin = indexerTable.getEntry("Blue V Min");
        blueVMin.setDouble(60);
        blueVMax = indexerTable.getEntry("Blue V Max");
        blueVMax.setDouble(252);


        output = output_;
        hsvImage = new Mat();
        blueMask = new Mat();
        redMask =new Mat();
        outputImage = new Mat();
        nextMat = new Mat();
        currentMat = new Mat();
        erosionKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(7,7));
    }

    @Override
    public void process(Mat inputImage) {
      frameCounter += 1;
     
      Imgproc.cvtColor(inputImage, hsvImage, Imgproc.COLOR_BGR2HSV_FULL);
      
      Mat currentROI = hsvImage.submat(0, hsvImage.rows()-1, 0, hsvImage.cols()/2-1);
      Mat nextROI = hsvImage.submat(0, hsvImage.rows()-1, hsvImage.cols()/2, hsvImage.cols()-1);

      Core.inRange(currentROI, new Scalar(redHMin.getDouble(0), redSMin.getDouble(90), redVMin.getDouble(20)), 
          new Scalar(redHMax.getDouble(30), redSMax.getDouble(250), redVMax.getDouble(240)), redMask);
      Core.inRange(currentROI, new Scalar(blueHMin.getDouble(0), blueSMin.getDouble(50), blueVMin.getDouble(20)), 
          new Scalar(blueHMax.getDouble(30), blueSMax.getDouble(250), blueVMax.getDouble(240)), blueMask);
      IndexerData currentIndexer = new IndexerData();
      findCargo(currentROI, redMask, blueMask, currentIndexer);

      Core.inRange(nextROI, new Scalar(redHMin.getDouble(0), redSMin.getDouble(90), redVMin.getDouble(20)), 
          new Scalar(redHMax.getDouble(30), redSMax.getDouble(250), redVMax.getDouble(240)), redMask);
      Core.inRange(nextROI, new Scalar(blueHMin.getDouble(0), blueSMin.getDouble(50), blueVMin.getDouble(20)), 
         new Scalar(blueHMax.getDouble(30), blueSMax.getDouble(250), blueVMax.getDouble(240)), blueMask);
      IndexerData nextIndexer = new IndexerData();
      findCargo(nextROI, redMask, blueMask, nextIndexer);

      String msg = String.format("Indexer: Current A %f, Next A %f", currentIndexer.area, nextIndexer.area);
      System.out.println(msg);

      double totalCargo = 0;
      if (currentIndexer.area > 0) {
        totalCargo = totalCargo + 1;
        if (currentIndexer.isRed == true) {
          current.setDouble(1);
        }
        else {
          current.setDouble(2);
        }
      }
      else {
        current.setDouble(0);
      }
      
      if (nextIndexer.area > 0) {
        totalCargo = totalCargo + 1;
        if (nextIndexer.isRed == true) {
          next.setDouble(1);
        }
        else {
          next.setDouble(2);
        }
      }
      else {
        next.setDouble(0);
      }

      total.setDouble(totalCargo);

      // TODO: Draw on input image for debugging?

      output.putFrame(inputImage);
      
      /* if (frameCounter%20 == 0){
      String fileName = String.format( "/media/usb_key/indexer_image_%d.jpg", frameCounter);
      if (Imgcodecs.imwrite(fileName, inputImage) == false){
      System.out.println("failed");
      }
      else {
        System.out.println("Success");
        }
      }*/
     }

  void findCargo(Mat inputImage, Mat redMask, Mat blueMask, IndexerData indexerData){

   double redCount = countPixels(redMask);
   double blueCount = countPixels(blueMask);

  indexerData.clear();
   
    if (redCount > blueCount && redCount > 30) {
      indexerData.area = redCount;
      indexerData.isRed = true;
    }
    else if (blueCount > 30) {
      indexerData.area = blueCount;
      indexerData.isRed = false;
    }
  }

  public double countPixels(Mat img) {
    double pixelCounter = 0;
    for (int row = 0; row < img.rows(); row = row + 1){
      for (int col = 0; col < img.cols(); col = col + 1) {
        if (img.get(row, col)[0] > 0){
          pixelCounter = pixelCounter + 1;
        }
      }
    }
    return pixelCounter;
  }
}
