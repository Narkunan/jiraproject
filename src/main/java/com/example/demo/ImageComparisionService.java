package com.example.demo;

import nu.pattern.OpenCV;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import javax.imageio.ImageIO;
import java.awt.AlphaComposite;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Service
public class ImageComparisionService {

    static {
        // Loads the OpenCV native library bundled by org.openpnp:opencv.
        OpenCV.loadLocally();
    }

    private static final int CANVAS_WIDTH = 1600;
    private static final int CANVAS_HEIGHT = 900;
    private static final int HEADER_HEIGHT = 70;
    private static final int PANEL_WIDTH = 790;
    private static final int PANEL_HEIGHT = 810;
    private static final int RIGHT_PANEL_X = 810;

    // Tune these values depending on how sensitive the comparison should be.
    private static final double DIFFERENCE_THRESHOLD = 25.0;
    private static final double MIN_CONTOUR_AREA = 20.0;

    /**
     * Main method used by the controller.
     * Returns a side-by-side PNG with differences highlighted on both panels.
     */
    public byte[] compareImages(
            MultipartFile expectedImage,
            MultipartFile actualImage) throws IOException {

        BufferedImage expected = readImage(expectedImage, "expected");
        BufferedImage actual = readImage(actualImage, "actual");

        List<Rectangle> differences = detectDifferences(expected, actual);

        return createAnnotatedComparisonImage(expected, actual, differences);
    }

    private BufferedImage readImage(
            MultipartFile file,
            String imageName) throws IOException {

        BufferedImage image = ImageIO.read(file.getInputStream());

        if (image == null) {
            throw new ResponseStatusException(
                    HttpStatus.BAD_REQUEST,
                    "Invalid " + imageName + " image. Upload a PNG or JPEG image."
            );
        }

        return image;
    }

    /**
     * Compares the actual screenshot against the expected screenshot.
     *
     * If dimensions differ, the actual screenshot is resized to the expected
     * dimensions for comparison only. The original images are still used in
     * the final side-by-side output.
     */
    private List<Rectangle> detectDifferences(
            BufferedImage expected,
            BufferedImage actual) {

        Mat expectedMat = null;
        Mat actualMat = null;
        Mat comparisonActual = new Mat();
        Mat expectedBlurred = new Mat();
        Mat actualBlurred = new Mat();
        Mat difference = new Mat();
        Mat grayDifference = new Mat();
        Mat binaryMask = new Mat();
        Mat hierarchy = new Mat();
        Mat openKernel = null;
        Mat mergeKernel = null;

        List<MatOfPoint> contours = new ArrayList<>();
        List<Rectangle> rectangles = new ArrayList<>();

        try {
            expectedMat = bufferedImageToMat(expected);
            actualMat = bufferedImageToMat(actual);

            // Compare at the expected screenshot's dimensions.
            if (actual.getWidth() != expected.getWidth()
                    || actual.getHeight() != expected.getHeight()) {

                Imgproc.resize(
                        actualMat,
                        comparisonActual,
                        new Size(expected.getWidth(), expected.getHeight()),
                        0,
                        0,
                        Imgproc.INTER_AREA
                );
            } else {
                actualMat.copyTo(comparisonActual);
            }

            // A small blur reduces noise caused by anti-aliasing.
            Imgproc.GaussianBlur(
                    expectedMat,
                    expectedBlurred,
                    new Size(3, 3),
                    0
            );

            Imgproc.GaussianBlur(
                    comparisonActual,
                    actualBlurred,
                    new Size(3, 3),
                    0
            );

            // Pixel-level comparison.
            Core.absdiff(expectedBlurred, actualBlurred, difference);

            Imgproc.cvtColor(
                    difference,
                    grayDifference,
                    Imgproc.COLOR_BGR2GRAY
            );

            // Ignore small colour/pixel variations below the threshold.
            Imgproc.threshold(
                    grayDifference,
                    binaryMask,
                    DIFFERENCE_THRESHOLD,
                    255,
                    Imgproc.THRESH_BINARY
            );

            // Remove isolated noise.
            openKernel = Imgproc.getStructuringElement(
                    Imgproc.MORPH_RECT,
                    new Size(3, 3)
            );

            Imgproc.morphologyEx(
                    binaryMask,
                    binaryMask,
                    Imgproc.MORPH_OPEN,
                    openKernel
            );

            // Join nearby changed pixels into more useful UI difference regions.
            mergeKernel = Imgproc.getStructuringElement(
                    Imgproc.MORPH_RECT,
                    new Size(5, 5)
            );

            Imgproc.dilate(
                    binaryMask,
                    binaryMask,
                    mergeKernel,
                    new Point(-1, -1),
                    2
            );

            Imgproc.findContours(
                    binaryMask,
                    contours,
                    hierarchy,
                    Imgproc.RETR_EXTERNAL,
                    Imgproc.CHAIN_APPROX_SIMPLE
            );

            for (MatOfPoint contour : contours) {
                double area = Imgproc.contourArea(contour);

                if (area < MIN_CONTOUR_AREA) {
                    continue;
                }

                Rect rect = Imgproc.boundingRect(contour);

                // A small padding makes the annotation easier to see.
                int padding = 3;
                int x = Math.max(0, rect.x - padding);
                int y = Math.max(0, rect.y - padding);
                int right = Math.min(
                        expected.getWidth(),
                        rect.x + rect.width + padding
                );
                int bottom = Math.min(
                        expected.getHeight(),
                        rect.y + rect.height + padding
                );

                rectangles.add(new Rectangle(
                        x,
                        y,
                        Math.max(1, right - x),
                        Math.max(1, bottom - y)
                ));
            }

            return rectangles;

        } finally {
            if (expectedMat != null) expectedMat.release();
            if (actualMat != null) actualMat.release();
            comparisonActual.release();
            expectedBlurred.release();
            actualBlurred.release();
            difference.release();
            grayDifference.release();
            binaryMask.release();
            hierarchy.release();

            if (openKernel != null) openKernel.release();
            if (mergeKernel != null) mergeKernel.release();

            for (MatOfPoint contour : contours) {
                contour.release();
            }
        }
    }

    /**
     * Builds the same side-by-side layout as the previous implementation.
     * Each detected region is highlighted on both the expected and actual
     * screenshot so the corresponding areas can be compared visually.
     */
    private byte[] createAnnotatedComparisonImage(
            BufferedImage expected,
            BufferedImage actual,
            List<Rectangle> differences) throws IOException {

        BufferedImage combined = new BufferedImage(
                CANVAS_WIDTH,
                CANVAS_HEIGHT,
                BufferedImage.TYPE_INT_RGB
        );

        Graphics2D graphics = combined.createGraphics();

        try {
            graphics.setRenderingHint(
                    RenderingHints.KEY_INTERPOLATION,
                    RenderingHints.VALUE_INTERPOLATION_BICUBIC
            );

            graphics.setRenderingHint(
                    RenderingHints.KEY_RENDERING,
                    RenderingHints.VALUE_RENDER_QUALITY
            );

            graphics.setColor(Color.WHITE);
            graphics.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

            // Header.
            graphics.setColor(new Color(35, 45, 60));
            graphics.fillRect(0, 0, CANVAS_WIDTH, HEADER_HEIGHT);

            graphics.setFont(new Font("Arial", Font.BOLD, 28));
            graphics.setColor(Color.WHITE);
            graphics.drawString("EXPECTED IMAGE", 280, 45);
            graphics.drawString("ACTUAL IMAGE", 1080, 45);

            // Draw the original images without modifying their content.
            Rectangle expectedBounds = drawImageFit(
                    graphics,
                    expected,
                    0,
                    HEADER_HEIGHT,
                    PANEL_WIDTH,
                    PANEL_HEIGHT
            );

            Rectangle actualBounds = drawImageFit(
                    graphics,
                    actual,
                    RIGHT_PANEL_X,
                    HEADER_HEIGHT,
                    PANEL_WIDTH,
                    PANEL_HEIGHT
            );

            // Draw the same detected comparison regions on both screenshots.
            for (Rectangle difference : differences) {
                drawDifference(
                        graphics,
                        difference,
                        expected.getWidth(),
                        expected.getHeight(),
                        expectedBounds
                );

                drawDifference(
                        graphics,
                        difference,
                        expected.getWidth(),
                        expected.getHeight(),
                        actualBounds
                );
            }

            // Divider between screenshots.
            graphics.setColor(Color.GRAY);
            graphics.setStroke(new BasicStroke(2));
            graphics.drawLine(
                    800,
                    HEADER_HEIGHT,
                    800,
                    CANVAS_HEIGHT
            );

        } finally {
            graphics.dispose();
        }

        ByteArrayOutputStream output = new ByteArrayOutputStream();
        ImageIO.write(combined, "png", output);
        return output.toByteArray();
    }

    private Rectangle drawImageFit(
            Graphics2D graphics,
            BufferedImage image,
            int x,
            int y,
            int maxWidth,
            int maxHeight) {

        double scale = Math.min(
                (double) maxWidth / image.getWidth(),
                (double) maxHeight / image.getHeight()
        );

        int width = (int) Math.round(image.getWidth() * scale);
        int height = (int) Math.round(image.getHeight() * scale);

        int drawX = x + (maxWidth - width) / 2;
        int drawY = y + (maxHeight - height) / 2;

        graphics.drawImage(image, drawX, drawY, width, height, null);

        return new Rectangle(drawX, drawY, width, height);
    }

    private void drawDifference(
            Graphics2D graphics,
            Rectangle difference,
            int comparisonWidth,
            int comparisonHeight,
            Rectangle imageBounds) {

        double scaleX = (double) imageBounds.width / comparisonWidth;
        double scaleY = (double) imageBounds.height / comparisonHeight;

        int x = imageBounds.x + (int) Math.round(difference.x * scaleX);
        int y = imageBounds.y + (int) Math.round(difference.y * scaleY);
        int width = Math.max(1, (int) Math.round(difference.width * scaleX));
        int height = Math.max(1, (int) Math.round(difference.height * scaleY));

        // Subtle transparent red fill.
        java.awt.Composite oldComposite = graphics.getComposite();

        graphics.setComposite(
                AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 0.22f)
        );
        graphics.setColor(Color.RED);
        graphics.fillRect(x, y, width, height);

        graphics.setComposite(oldComposite);

        // Red outline.
        graphics.setColor(new Color(230, 0, 0));
        graphics.setStroke(new BasicStroke(2.0f));
        graphics.drawRect(x, y, width, height);
    }

    /**
     * OpenCV expects a continuous 8-bit BGR image.
     */
    private Mat bufferedImageToMat(BufferedImage source) {

        BufferedImage bgr = new BufferedImage(
                source.getWidth(),
                source.getHeight(),
                BufferedImage.TYPE_3BYTE_BGR
        );

        Graphics2D graphics = bgr.createGraphics();
        graphics.drawImage(source, 0, 0, null);
        graphics.dispose();

        byte[] pixels = ((DataBufferByte) bgr.getRaster()
                .getDataBuffer()).getData();

        Mat mat = new Mat(
                bgr.getHeight(),
                bgr.getWidth(),
                org.opencv.core.CvType.CV_8UC3
        );

        mat.put(0, 0, pixels);
        return mat;
    }
}
