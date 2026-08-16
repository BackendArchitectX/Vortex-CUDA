package com.vortex.server.api;

import com.vortex.server.config.VortexLimitsProperties;
import com.vortex.server.config.VortexSecurityProperties;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;

import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.Locale;
import java.util.UUID;
import java.util.regex.Pattern;

@Component
@Order(Ordered.HIGHEST_PRECEDENCE + 10)
public final class RequestGuardFilter
        extends OncePerRequestFilter {

    public static final String REQUEST_ID_HEADER =
            "X-Request-ID";

    public static final String API_KEY_HEADER =
            "X-Vortex-Api-Key";

    private static final String MDC_REQUEST_ID =
            "requestId";

    private static final Pattern SAFE_REQUEST_ID =
            Pattern.compile(
                    "[A-Za-z0-9._-]{1,128}"
            );

    private static final Logger LOGGER =
            LoggerFactory.getLogger(
                    RequestGuardFilter.class
            );

    private final VortexSecurityProperties security;
    private final VortexLimitsProperties limits;

    public RequestGuardFilter(
            VortexSecurityProperties security,
            VortexLimitsProperties limits
    ) {
        this.security = security;
        this.limits = limits;
    }

    @Override
    protected void doFilterInternal(
            HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain
    ) throws ServletException, IOException {

        String requestId =
                resolveRequestId(
                        request
                );

        long startNanos =
                System.nanoTime();

        MDC.put(
                MDC_REQUEST_ID,
                requestId
        );

        response.setHeader(
                REQUEST_ID_HEADER,
                requestId
        );

        try {
            if (isApiRequest(request)) {
                if (requestTooLarge(request)) {
                    writeProblem(
                            response,
                            HttpServletResponse
                                    .SC_REQUEST_ENTITY_TOO_LARGE,
                            "PAYLOAD_TOO_LARGE",
                            "Payload Too Large",
                            "Request body exceeds configured maximum.",
                            requestId
                    );

                    return;
                }

                if (!authorized(request)) {
                    response.setHeader(
                            "WWW-Authenticate",
                            "VortexApiKey"
                    );

                    writeProblem(
                            response,
                            HttpServletResponse
                                    .SC_UNAUTHORIZED,
                            "UNAUTHORIZED",
                            "Unauthorized",
                            "A valid X-Vortex-Api-Key header is required.",
                            requestId
                    );

                    return;
                }
            }

            filterChain.doFilter(
                    request,
                    response
            );
        } finally {
            long elapsedMicros =
                    (
                            System.nanoTime() -
                                    startNanos
                    ) /
                            1_000L;

            LOGGER.info(
                    "request method={} path={} status={} durationMicros={}",
                    request.getMethod(),
                    request.getRequestURI(),
                    response.getStatus(),
                    elapsedMicros
            );

            MDC.remove(
                    MDC_REQUEST_ID
            );
        }
    }

    private boolean requestTooLarge(
            HttpServletRequest request
    ) {
        long contentLength =
                request.getContentLengthLong();

        return contentLength >= 0 &&
                contentLength >
                        limits.maxRequestBytes();
    }

    private boolean authorized(
            HttpServletRequest request
    ) {
        if (!security.apiKeyEnabled()) {
            return true;
        }

        String provided =
                request.getHeader(
                        API_KEY_HEADER
                );

        if (provided == null) {
            return false;
        }

        byte[] expectedBytes =
                security.apiKey()
                        .getBytes(
                                StandardCharsets.UTF_8
                        );

        byte[] providedBytes =
                provided.getBytes(
                        StandardCharsets.UTF_8
                );

        return MessageDigest.isEqual(
                expectedBytes,
                providedBytes
        );
    }

    private static boolean isApiRequest(
            HttpServletRequest request
    ) {
        return request.getRequestURI()
                .startsWith(
                        "/api/"
                );
    }

    private static String resolveRequestId(
            HttpServletRequest request
    ) {
        String supplied =
                request.getHeader(
                        REQUEST_ID_HEADER
                );

        if (supplied != null &&
                SAFE_REQUEST_ID
                        .matcher(supplied)
                        .matches()) {

            return supplied;
        }

        return UUID.randomUUID()
                .toString();
    }

    private static void writeProblem(
            HttpServletResponse response,
            int status,
            String code,
            String title,
            String detail,
            String requestId
    ) throws IOException {

        response.setStatus(status);
        response.setCharacterEncoding(
                StandardCharsets.UTF_8.name()
        );
        response.setContentType(
                "application/problem+json"
        );

        String type =
                "urn:vortex:error:" +
                        code.toLowerCase(
                                Locale.ROOT
                        );

        String json =
                "{" +
                        "\"type\":\"" +
                        type +
                        "\"," +
                        "\"title\":\"" +
                        title +
                        "\"," +
                        "\"status\":" +
                        status +
                        "," +
                        "\"detail\":\"" +
                        detail +
                        "\"," +
                        "\"code\":\"" +
                        code +
                        "\"," +
                        "\"requestId\":\"" +
                        requestId +
                        "\"," +
                        "\"timestamp\":\"" +
                        Instant.now() +
                        "\"" +
                        "}";

        response.getWriter()
                .write(json);
    }
}
