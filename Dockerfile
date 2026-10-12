
FROM maven:4.0.0-rc-7-eclipse-temurin-25-noble

WORKDIR /app

EXPOSE 8080

CMD ["mvn", "spring-boot:run"]

