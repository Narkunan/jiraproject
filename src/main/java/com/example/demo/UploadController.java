
package com.example.demo;

import java.io.IOException;
import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
public class UploadController {

    private final BedrockService bedrockService;

    public UploadController(BedrockService bedrockService) {
        this.bedrockService = bedrockService;
    }

    @PostMapping("/upload")
    public ResponseEntity<String> uploadFiles(@RequestParam("expected_image") MultipartFile expectedImage, @RequestParam("actual_image") MultipartFile actualImage) throws IOException {
        String prompt = "";

        if (expectedImage.isEmpty() || actualImage.isEmpty()) {
            return ResponseEntity.badRequest()
                    .body("Please upload both expected and actual images.");
        }

        prompt = "assume you are QA engineer in a company first image is expected image and second image is actual image, compare both images and give me a report in detail with all the differences and similarities between both images, also give me a score out of 100 based on the comparison, now answer this prompt: " + prompt;

        String result = bedrockService.askWithImages(prompt, List.of(expectedImage, actualImage));

        return ResponseEntity.ok(result);
    }
}
