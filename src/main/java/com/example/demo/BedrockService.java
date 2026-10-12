
package com.example.demo;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import software.amazon.awssdk.core.SdkBytes;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.bedrockruntime.BedrockRuntimeClient;
import software.amazon.awssdk.services.bedrockruntime.model.*;

@Service
public class BedrockService {

    private final BedrockRuntimeClient client;
    private final String modelId;

    public BedrockService(
            @Value("${aws.region}") String region,
            @Value("${aws.bedrock.model-id}") String modelId) {

        this.client = BedrockRuntimeClient.builder()
                .region(Region.of(region))
                .build();

        this.modelId = modelId;
    }

    public String askWithImages(
            String prompt,
            List<MultipartFile> images) throws IOException {

        List<ContentBlock> contentBlocks = new ArrayList<>();

        // Add the user's prompt.
        contentBlocks.add(ContentBlock.fromText(prompt));

        // Add each image to the same message.
        for (MultipartFile file : images) {
            if (file.isEmpty()) {
                continue;
            }

            ImageFormat format = switch (file.getContentType() == null
                    ? "" : file.getContentType()) {
                case "image/jpeg" -> ImageFormat.JPEG;
                case "image/png" -> ImageFormat.PNG;
                default -> throw new IllegalArgumentException(
                        "Only JPEG and PNG images are supported.");
            };

            ImageBlock image = ImageBlock.builder()
                    .format(format)
                    .source(ImageSource.builder()
                            .bytes(SdkBytes.fromByteArray(file.getBytes()))
                            .build())
                    .build();

            contentBlocks.add(
                    ContentBlock.builder()
                            .image(image)
                            .build()
            );
        }

        Message message = Message.builder()
                .role(ConversationRole.USER)
                .content(contentBlocks)
                .build();

        ConverseResponse response = client.converse(request -> request
                .modelId(modelId)
                .messages(message)
                .inferenceConfig(config -> config
                        .maxTokens(500)
                        .temperature(0.3F)));

        return response.output()
                .message()
                .content()
                .stream()
                .filter(block -> block.text() != null)
                .map(ContentBlock::text)
                .reduce("", String::concat);
    }
}
