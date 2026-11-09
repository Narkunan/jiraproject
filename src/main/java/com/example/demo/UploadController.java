package com.example.demo;

import com.example.demo.ImageComparisionService;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.lang.InterruptedException;
import java.io.IOException;

@RestController
@RequestMapping("/api/images")
public class UploadController {

    private final ImageComparisionService imageComparisonService;

    public UploadController(
            ImageComparisionService imageComparisonService) {
        this.imageComparisonService = imageComparisonService;
    }

    @PostMapping(
            value = "/upload",
            consumes = MediaType.MULTIPART_FORM_DATA_VALUE,
            produces = MediaType.IMAGE_PNG_VALUE
    )
    public ResponseEntity<byte[]> uploadFiles(
            @RequestParam("expected_image") MultipartFile expectedImage,
            @RequestParam("actual_image") MultipartFile actualImage)
            throws IOException, InterruptedException {

        if (expectedImage.isEmpty() || actualImage.isEmpty()) {
            return ResponseEntity.badRequest().build();
        }

        byte[] result = imageComparisonService.compareImages(
                expectedImage,
                actualImage
        );

        return ResponseEntity.ok()
                .contentType(MediaType.parseMediaType("image/webp"))
                .header(
                        "Content-Disposition",
                        "inline; filename=\"comparison.webp\""
                )
                .body(result);
    }
}