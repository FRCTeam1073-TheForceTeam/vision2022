import edu.wpi.first.vision.VisionPipeline;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableEntry;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.features2d.SimpleBlobDetector;
import org.opencv.imgproc.*;
import edu.wpi.cscore.CvSource;
import edu.wpi.first.wpilibj.shuffleboard.ShuffleboardLayout;
import edu.wpi.first.wpilibj.shuffleboard.ShuffleboardTab;
import edu.wpi.first.wpilibj.shuffleboard.ShuffleboardContainer;
    
public class CargoTracker implements VisionPipeline {
    public int frameCounter;
    private NetworkTableInstance nti;
    private NetworkTable cargoTable;
    private NetworkTableEntry cargoX;
    private NetworkTableEntry cargoY;
    private NetworkTableEntry cargoArea;
    private NetworkTableEntry cargoHMin;
    private NetworkTableEntry cargoHMax;
    private NetworkTableEntry cargoSMin;
    private NetworkTableEntry cargoSMax;
    private NetworkTableEntry cargoVMin;
    private NetworkTableEntry cargoVMax;
    private CvSource output;
    private Mat hsvImage;
    private Mat maskImage;
    private Mat outputImage;

    //private SimpleBlobDetectorParams blobParam

    public CargoTracker(NetworkTableInstance ntinst, CvSource output_){
        nti = ntinst;
        cargoTable = nti.getTable("CARGO");
        cargoX = cargoTable.getEntry("Cargo X");
        cargoX.setDouble(0);
        cargoY = cargoTable.getEntry("Cargo Y");
        cargoY.setDouble(0);
        cargoArea = cargoTable.getEntry("Cargo Area");
        cargoArea.setDouble(0);
        cargoHMin = cargoTable.getEntry("H Min");
        cargoHMin.setDouble(30);
        cargoHMax = cargoTable.getEntry("H Max");
        cargoHMax.setDouble(50);
        cargoSMin = cargoTable.getEntry("S Min");
        cargoSMin.setDouble(50);
        cargoSMax = cargoTable.getEntry("S Max");
        cargoSMax.setDouble(250);
        cargoVMin = cargoTable.getEntry("V Min");
        cargoVMin.setDouble(20);
        cargoVMax = cargoTable.getEntry("V Max");
        cargoVMax.setDouble(240);


        output = output_;
        hsvImage = new Mat();
        maskImage = new Mat();
        outputImage = new Mat();
    }

    @Override
    public void process(Mat inputImage) {
      frameCounter += 1;
     
      Imgproc.cvtColor(inputImage, hsvImage, Imgproc.COLOR_BGR2HSV_FULL);
      Core.inRange(hsvImage, new Scalar(cargoHMin.getDouble(30), cargoSMin.getDouble(50), cargoVMin.getDouble(20)), new Scalar(cargoHMax.getDouble(50), cargoSMax.getDouble(250), cargoVMax.getDouble(240)), maskImage);
      outputImage.setTo(new Scalar(0,0,0));
      Core.bitwise_and(inputImage, inputImage, outputImage, maskImage);
     // Imgproc.Sobel(inputImage, outputImage, -1, 1, 1);
      //Imgproc.line(outputImage, new Point(0, outputImage.rows()/2), new Point(outputImage.cols()-1, outputImage.rows()/2), new Scalar(0, 0, 255));
      output.putFrame(outputImage);
    }
  }

