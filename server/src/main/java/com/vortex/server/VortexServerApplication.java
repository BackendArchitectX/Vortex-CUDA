package com.vortex.server;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication
@ConfigurationPropertiesScan
public class VortexServerApplication {

    public static void main(String[] args) {
        SpringApplication.run(
                VortexServerApplication.class,
                args
        );
    }
}
