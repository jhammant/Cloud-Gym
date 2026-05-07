package cloudgym.taxi

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import com.sun.net.httpserver.HttpExchange
import com.sun.net.httpserver.HttpHandler
import com.sun.net.httpserver.HttpServer
import com.fasterxml.jackson.module.kotlin.readValue
import lang.taxi.CompilationError
import lang.taxi.Compiler
import lang.taxi.sources.SourceCode
import java.net.InetSocketAddress
import java.util.concurrent.Executors

private val mapper = ObjectMapper().registerKotlinModule()

private data class ValidationResponse(
    val isValid: Boolean,
    val errorCount: Int,
    val warningCount: Int,
    val errors: List<ErrorRecord>,
)

private data class ErrorRecord(
    val line: Int,
    val char: Int,
    val severity: String,
    val detailMessage: String,
    val errorCode: String?,
    val sourceName: String?,
)

private fun CompilationError.toRecord(): ErrorRecord =
    ErrorRecord(
        line = line,
        char = char,
        severity = severity.label,
        detailMessage = detailMessage,
        errorCode = errorCode,
        sourceName = sourceName,
    )

private fun toResponse(msgs: List<CompilationError>): ValidationResponse {
    val errs = msgs.filter { it.severity.label.equals("Error", ignoreCase = true) }
    val warns = msgs.filter { it.severity.label.equals("Warning", ignoreCase = true) }
    return ValidationResponse(
        isValid = errs.isEmpty(),
        errorCount = errs.size,
        warningCount = warns.size,
        errors = msgs.map { it.toRecord() },
    )
}

private fun validate(source: String, sourceName: String): ValidationResponse =
    toResponse(Compiler(source, sourceName = sourceName).validate())

private data class SourceInput(val name: String, val content: String)
private data class MultiRequest(val sources: List<SourceInput>)

private fun validateMulti(sources: List<SourceInput>): ValidationResponse {
    if (sources.isEmpty()) {
        return ValidationResponse(isValid = false, errorCount = 1, warningCount = 0, errors = listOf(
            ErrorRecord(0, 0, "Error", "No sources provided", "EmptyRequest", null)))
    }
    val sourceCodes = sources.map { SourceCode(it.name, it.content) }
    return toResponse(Compiler(sourceCodes).validate())
}

private class ValidateHandler : HttpHandler {
    override fun handle(exchange: HttpExchange) {
        try {
            if (exchange.requestMethod != "POST") {
                exchange.sendResponseHeaders(405, -1); return
            }
            val source = exchange.requestBody.readBytes().toString(Charsets.UTF_8)
            val sourceName = exchange.requestHeaders.getFirst("X-Source-Name") ?: "input.taxi"
            val resp = try {
                validate(source, sourceName)
            } catch (e: Throwable) {
                ValidationResponse(
                    isValid = false,
                    errorCount = 1,
                    warningCount = 0,
                    errors = listOf(
                        ErrorRecord(
                            line = 0, char = 0, severity = "Error",
                            detailMessage = "Validator threw: ${e.javaClass.simpleName}: ${e.message}",
                            errorCode = "ValidatorThrew", sourceName = sourceName,
                        ),
                    ),
                )
            }
            val bytes = mapper.writeValueAsBytes(resp)
            exchange.responseHeaders.add("Content-Type", "application/json")
            exchange.sendResponseHeaders(200, bytes.size.toLong())
            exchange.responseBody.use { it.write(bytes) }
        } finally {
            exchange.close()
        }
    }
}

private class MultiValidateHandler : HttpHandler {
    override fun handle(exchange: HttpExchange) {
        try {
            if (exchange.requestMethod != "POST") {
                exchange.sendResponseHeaders(405, -1); return
            }
            val body = exchange.requestBody.readBytes().toString(Charsets.UTF_8)
            val resp = try {
                val req: MultiRequest = mapper.readValue(body)
                validateMulti(req.sources)
            } catch (e: Throwable) {
                ValidationResponse(
                    isValid = false,
                    errorCount = 1,
                    warningCount = 0,
                    errors = listOf(
                        ErrorRecord(
                            line = 0, char = 0, severity = "Error",
                            detailMessage = "Validator threw: ${e.javaClass.simpleName}: ${e.message}",
                            errorCode = "ValidatorThrew", sourceName = null,
                        ),
                    ),
                )
            }
            val bytes = mapper.writeValueAsBytes(resp)
            exchange.responseHeaders.add("Content-Type", "application/json")
            exchange.sendResponseHeaders(200, bytes.size.toLong())
            exchange.responseBody.use { it.write(bytes) }
        } finally {
            exchange.close()
        }
    }
}

private class HealthHandler : HttpHandler {
    override fun handle(exchange: HttpExchange) {
        val body = """{"status":"ok"}""".toByteArray()
        exchange.responseHeaders.add("Content-Type", "application/json")
        exchange.sendResponseHeaders(200, body.size.toLong())
        exchange.responseBody.use { it.write(body) }
    }
}

fun main(args: Array<String>) {
    val port = (System.getenv("PORT") ?: args.firstOrNull() ?: "9123").toInt()
    val server = HttpServer.create(InetSocketAddress("127.0.0.1", port), 0)
    server.createContext("/validate", ValidateHandler())
    server.createContext("/validate-multi", MultiValidateHandler())
    server.createContext("/health", HealthHandler())
    server.executor = Executors.newFixedThreadPool(8)
    server.start()
    println("taxi-validator-shim listening on http://127.0.0.1:$port")
    println("  POST /validate         body=Taxi source       header X-Source-Name (optional)")
    println("  POST /validate-multi   body={\"sources\":[{\"name\":...,\"content\":...}]}")
    println("  GET  /health")
}
