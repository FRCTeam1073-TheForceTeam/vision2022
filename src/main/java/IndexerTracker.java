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
    private NetworkTableEntry indexerX;
    private NetworkTableEntry indexerY;
    private NetworkTableEntry indexerArea;
    private NetworkTableEntry indexerHMin;
    private NetworkTableEntry indexerHMax;
    private NetworkTableEntry indexerSMin;
    private NetworkTableEntry indexerSMax;
    private NetworkTableEntry indexerVMin;
    private NetworkTableEntry indexerVMax;
    private CvSource output;
    private Mat hsvImage;
    private Mat maskImage;
    private Mat outputImage;
    private Mat erosionKernel;

    //private SimpleBlobDetectorParams blobParam

    public IndexerTracker(NetworkTableInstance ntinst, CvSource output_){
        nti = ntinst;
        indexerTable = nti.getTable("INDEXER");
        indexerX = indexerTable.getEntry("Indexer X");
        indexerX.setDouble(0);
        indexerY = indexerTable.getEntry("Indexer Y");
        indexerY.setDouble(0);
        indexerArea = indexerTable.getEntry("Indexer Area");
        indexerArea.setDouble(0);
        indexerHMin = indexerTable.getEntry("H Min");
        indexerHMin.setDouble(0);
        indexerHMax = indexerTable.getEntry("H Max");
        indexerHMax.setDouble(30);
        indexerSMin = indexerTable.getEntry("S Min");
        indexerSMin.setDouble(90);
        indexerSMax = indexerTable.getEntry("S Max");
        indexerSMax.setDouble(255);
        indexerVMin = indexerTable.getEntry("V Min");
        indexerVMin.setDouble(60);
        indexerVMax = indexerTable.getEntry("V Max");
        indexerVMax.setDouble(252);


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
      Core.inRange(hsvImage, new Scalar(indexerHMin.getDouble(0), indexerSMin.getDouble(50), indexerVMin.getDouble(20)), new Scalar(indexerHMax.getDouble(30), indexerSMax.getDouble(250), indexerVMax.getDouble(240)), maskImage);
      outputImage.setTo(new Scalar(0,0,0));
      Core.bitwise_and(inputImage, inputImage, outputImage, maskImage);
      /*
      red cargo: H max(30), H min(0), S max(255), S min(90), V max(252), V min(60)
      blue cargo: H max(165), H min(130), S max(255), S min(90), V max(252), V min(60)
      */

      // Erode the mask image to eliminate the little "noise" pixels
      Imgproc.erode(maskImage, maskImage, erosionKernel);

      // Set up to find contours in the mask image.
      List<MatOfPoint> contours = new ArrayList<>();
      Mat hierarchy = new Mat();
      Imgproc.findContours(maskImage, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

      double bestArea = 0.0;
      double bestX = 0.0;
      double bestY = 0.0;


      // Walk the list of contours that we found and draw the ones that meet certain criteria:
      for (int cidx=0; cidx < contours.size(); cidx++) {
        // Grab the bounding rectangle for contour:
        Rect bounds = Imgproc.boundingRect(contours.get(cidx));
        double aspecterr = Math.abs(1.0 - (double)bounds.width/(double)bounds.height);
        Imgproc.rectangle(outputImage, bounds.br(), bounds.tl(), new Scalar(255,0,0));

        // Only draw contours that have nearly square bounding boxes, and some minimal area... like round things.
        if (bounds.area() > 32 && aspecterr < 0.3) {
          // Now we know it has non-trivial size and is closer to square/round:

          Imgproc.drawContours(outputImage, contours, cidx, new Scalar(0, 255, 0));
         if (bounds.area() > bestArea){
           bestArea = bounds.area();
           bestX = bounds.x;
           bestY = bounds.y;
         }
        }
      }
      //sends our best answer if found
      if (bestArea > 0.0){
          indexerX.setDouble(bestX);
          indexerY.setDouble(bestY);
          indexerArea.setDouble(bestArea);
      }
      else {
        indexerArea.setDouble(0.0);

      }


     // Imgproc.Sobel(inputImage, outputImage, -1, 1, 1);
      //Imgproc.line(outputImage, new Point(0, outputImage.rows()/2), new Point(outputImage.cols()-1, outputImage.rows()/2), new Scalar(0, 0, 255));

      output.putFrame(outputImage);
    }

}
