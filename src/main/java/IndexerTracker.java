import edu.wpi.first.vision.VisionPipeline;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableEntry;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.MatOfPoint;
import org.opencv.features2d.SimpleBlobDetector;
import org.opencv.imgproc.*;
import edu.wpi.cscore.CvSource;

public class IndexerTracker implements VisionPipeline {
    public int frameCounter;
    private NetworkTableInstance nti;
    private NetworkTable indexerTable;
    private NetworkTableEntry redIndexerX;
    private NetworkTableEntry redIndexerY;
    private NetworkTableEntry redIndexerArea;
    private NetworkTableEntry blueIndexerX;
    private NetworkTableEntry blueIndexerY;
    private NetworkTableEntry blueIndexerArea;
    private NetworkTableEntry isRed;
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
    private Mat maskImage;
    private Mat outputImage;
    private Mat erosionKernel;

    public class IndexerData{
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
        redIndexerX = indexerTable.getEntry("Red Indexer X");
        redIndexerX.setDouble(0);
        redIndexerY = indexerTable.getEntry("Red Indexer Y");
        redIndexerY.setDouble(0);
        redIndexerArea = indexerTable.getEntry("Red Indexer Area");
        redIndexerArea.setDouble(0);
        blueIndexerX = indexerTable.getEntry("Blue Indexer X");
        blueIndexerX.setDouble(0);
        blueIndexerY = indexerTable.getEntry("Blue Indexer Y");
        blueIndexerY.setDouble(0);
        blueIndexerArea = indexerTable.getEntry("Blue Indexer Area");
        blueIndexerArea.setDouble(0);
        isRed = indexerTable.getEntry("Is cargo Red");
        isRed.setBoolean(true);
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
        maskImage = new Mat();
        outputImage = new Mat();
        erosionKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(7,7));
    }

    @Override
    public void process(Mat inputImage) {
      frameCounter += 1;
     
      Imgproc.cvtColor(inputImage, hsvImage, Imgproc.COLOR_BGR2HSV_FULL);

     Core.inRange(hsvImage, new Scalar(redHMin.getDouble(0), redSMin.getDouble(90), redVMin.getDouble(20)), 
          new Scalar(redHMax.getDouble(30), redSMax.getDouble(250), redVMax.getDouble(240)), maskImage);
      IndexerData redIndexer = new IndexerData();
      findCargo(inputImage, maskImage, redIndexer);

      Core.inRange(hsvImage, new Scalar(blueHMin.getDouble(0), blueSMin.getDouble(50), blueVMin.getDouble(20)), 
         new Scalar(blueHMax.getDouble(30), blueSMax.getDouble(250), blueVMax.getDouble(240)), maskImage);
      IndexerData blueIndexer = new IndexerData();
      findCargo(inputImage, maskImage, blueIndexer);

      //sends our best answer if found
      if (redIndexer.area > 0.0){
          redIndexerX.setDouble(redIndexer.x);
          redIndexerY.setDouble(redIndexer.y);
          redIndexerArea.setDouble(redIndexer.area);
          isRed.setBoolean(true);
          Imgproc.rectangle(inputImage, new Point(redIndexer.x - redIndexer.width/2.0, redIndexer.y - redIndexer.height/2.0), 
              new Point(redIndexer.x + redIndexer.width/2.0, redIndexer.y + redIndexer.height/2.0), new Scalar(0,0,255), 3);
      }
      else {
        redIndexerArea.setDouble(0.0);
      }
      
      if (blueIndexer.area > 0.0){
          blueIndexerX.setDouble(blueIndexer.x);
          blueIndexerY.setDouble(blueIndexer.y);
          blueIndexerArea.setDouble(blueIndexer.area);
          isRed.setBoolean(false);
          Imgproc.rectangle(inputImage, new Point(blueIndexer.x - blueIndexer.width/2.0, blueIndexer.y - blueIndexer.height/2.0), 
              new Point(blueIndexer.x + blueIndexer.width/2.0, blueIndexer.y + blueIndexer.height/2.0), new Scalar(255,0,0), 3);
      }
      else {
        blueIndexerArea.setDouble(0.0);
      }

     // Imgproc.Sobel(inputImage, outputImage, -1, 1, 1);
      //Imgproc.line(outputImage, new Point(0, outputImage.rows()/2), new Point(outputImage.cols()-1, outputImage.rows()/2), new Scalar(0, 0, 255));

      output.putFrame(inputImage);
     }

  void findCargo(Mat inputImage, Mat maskImage, IndexerData indexerData){
    outputImage.setTo(new Scalar(0,0,0));
    Core.bitwise_and(inputImage, inputImage, outputImage, maskImage);
    /*
    red cargo: H max(30), H min(0), S max(255), S min(90), V max(252), V min(60)
    blue cargo: H max(165), H min(130), S max(255), S min(90), V max(252), V min(60)
    */

    // Erode the mask image to eliminate the little "noise" pixels
    //Imgproc.erode(maskImage, maskImage, erosionKernel);

    // Set up to find contours in the mask image.
    List<MatOfPoint> contours = new ArrayList<>();
    Mat hierarchy = new Mat();
    Imgproc.findContours(maskImage, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

    double bestArea = 0.0;
    double bestX = 0.0;
    double bestY = 0.0;
    double bestWidth = 0.0;
    double bestHeight = 0.0;


    // Walk the list of contours that we found and draw the ones that meet certain criteria:
    for (int cidx=0; cidx < contours.size(); cidx++) {
      // Grab the bounding rectangle for contour:
      Rect bounds = Imgproc.boundingRect(contours.get(cidx));
      double aspecterr = Math.abs(1.0 - (double)bounds.width/(double)bounds.height);
    //  Imgproc.rectangle(outputImage, bounds.br(), bounds.tl(), new Scalar(255,0,0));

      // Only draw contours that have nearly square bounding boxes, and some minimal area... like round things.
      if (bounds.area() > 32 && aspecterr < 0.3) {
        // Now we know it has non-trivial size and is closer to square/round:

        // Imgproc.drawContours(outputImage, contours, cidx, new Scalar(0, 255, 0));
        if (bounds.area() > bestArea){
          bestArea = bounds.area();
          bestX = bounds.x + bounds.width/2.0;
          bestY = bounds.y + bounds.height/2.0;
          bestWidth = bounds.width;
          bestHeight = bounds.height;
        }
      }
    }
    //sends our best answer if found
    indexerData.x = bestX;
    indexerData.y = bestY;
    indexerData.area = bestArea;
    indexerData.width = bestWidth;
    indexerData.height = bestHeight;
    }
  }

