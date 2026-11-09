"""Contract tests for the public CiteRAG Lite demo entrypoint.

The demo is served publicly on the Northflank free plan, so these tests pin the
behaviour that matters for that exposure: read-only retrieval over the shipped
documentation, no bearer-token requirement, a closed configuration guard, and
no internal detail leaking through error paths.
"""
import json
import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from starlette.testclient import TestClient

from citerag import demo_web
from citerag.cli import ingest

GUARDED_ENV = (
    "CITERAG_DEMO_PUBLIC",
    "CITERAG_DEMO_ALLOW_REINDEX",
    "CITERAG_PRODUCTION",
    "CITERAG_REQUIRE_AUTH",
    "CITERAG_API_TOKEN",
)


class DemoWebContractTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        for name in GUARDED_ENV:
            os.environ.pop(name, None)

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db = pathlib.Path(temporary.name) / "index.sqlite3"
        self.documents = ingest(self.db, demo_web.DOCS)
        self.assertGreater(self.documents, 0)

        database = patch.object(demo_web, "DB", self.db)
        database.start()
        self.addCleanup(database.stop)

        reindex_state = patch.object(demo_web, "_last_reindex", 0.0)
        reindex_state.start()
        self.addCleanup(reindex_state.stop)

        self.client = TestClient(demo_web.app)

    def test_health_reports_fts5_backend_and_read_only_state(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["retrieval_backend"], "sqlite-fts5")
        self.assertEqual(payload["retrieval_index"], "present")
        self.assertEqual(payload["generation"], "not_configured")
        self.assertEqual(payload["native_opencode"], "not_configured")
        self.assertEqual(payload["reindex"], "disabled")
        self.assertFalse(payload["busy"])

    def test_home_serves_the_demo_banner(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("CiteRAG Lite demo", response.text)
        self.assertIn("SQLite FTS5", response.text)

    def test_adapters_expose_only_retrieval_only(self):
        response = self.client.get("/api/adapters")
        self.assertEqual(response.status_code, 200)
        adapters = {item["name"]: item for item in response.json()["adapters"]}
        self.assertEqual(set(adapters), {"retrieval-only", "omniroute", "opencode"})
        self.assertTrue(adapters["retrieval-only"]["available"])
        self.assertFalse(adapters["omniroute"]["available"])
        self.assertFalse(adapters["opencode"]["available"])

    def test_context_reports_the_indexed_document_count(self):
        response = self.client.get("/api/context")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["imported_documents"], self.documents)
        self.assertEqual(payload["workspace"], "demo")

    def test_ask_returns_a_cited_answer(self):
        response = self.client.post(
            "/api/ask",
            json={"question": "Which vector database and embedding model does this project use?", "adapter": "retrieval-only"},
        )
        self.assertEqual(response.status_code, 202)
        job_id = response.json()["job_id"]
        result = self.client.get("/api/jobs/" + job_id).json()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["mode"], "retrieval_only")
        self.assertTrue(result["citations_verified"])
        self.assertTrue(result["sources"])
        self.assertTrue(all(source.startswith("docs/") for source in result["sources"]))
        self.assertIn("Source:", result["answer"])

    def test_ask_rejects_invalid_payloads(self):
        for payload in (
            {"question": "   "},
            {"question": "x" * 4001},
            {"question": "valid", "adapter_config": {"api_key": "never"}},
        ):
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/api/ask", json=payload).status_code, 400)
        self.assertFalse(demo_web._generation.locked())

    def test_unavailable_adapters_are_refused_with_503(self):
        response = self.client.post("/api/ask", json={"question": "valid", "adapter": "omniroute"})
        self.assertEqual(response.status_code, 503)
        response = self.client.post("/api/ask", json={"question": "valid", "use_native": True})
        self.assertEqual(response.status_code, 503)

    def test_unknown_job_is_not_found(self):
        self.assertEqual(self.client.get("/api/jobs/unknown").status_code, 404)

    def test_single_flight_limit_returns_conflict(self):
        demo_web._generation.acquire()
        try:
            response = self.client.post("/api/ask", json={"question": "Which vector database does CiteRAG use?"})
        finally:
            demo_web._generation.release()
        self.assertEqual(response.status_code, 409)

    def test_job_failure_reports_the_exception_type_only(self):
        with patch.object(demo_web, "search", side_effect=RuntimeError("secret /etc/passwd")):
            response = self.client.post("/api/ask", json={"question": "Which vector database does CiteRAG use?"})
        self.assertEqual(response.status_code, 202)
        result = self.client.get("/api/jobs/" + response.json()["job_id"]).json()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "RuntimeError")
        self.assertNotIn("secret", json.dumps(result))

    def test_fts_query_metacharacters_do_not_break_retrieval(self):
        for question in ('"documents MATCH * OR', "a OR b NEAR(c)", "'; DROP TABLE documents; --"):
            with self.subTest(question=question):
                response = self.client.post("/api/ask", json={"question": question})
                self.assertEqual(response.status_code, 202)
                result = self.client.get("/api/jobs/" + response.json()["job_id"]).json()
                self.assertIn(result["status"], {"completed", "insufficient_evidence"})

    def test_reindex_is_disabled_for_the_public_demo(self):
        response = self.client.post("/api/reindex", json={})
        self.assertEqual(response.status_code, 403)
        self.assertIn("disabled", response.json()["error"])

    def test_reindex_requires_an_explicit_operator_opt_in(self):
        os.environ["CITERAG_DEMO_ALLOW_REINDEX"] = "1"
        response = self.client.post("/api/reindex", json={})
        self.assertEqual(response.status_code, 202)
        result = self.client.get("/api/jobs/" + response.json()["job_id"]).json()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["documents"], self.documents)
        limited = self.client.post("/api/reindex", json={})
        self.assertEqual(limited.status_code, 429)
        self.assertIn("Retry-After", limited.headers)

    def test_reindex_rejects_operator_supplied_sources(self):
        os.environ["CITERAG_DEMO_ALLOW_REINDEX"] = "1"
        response = self.client.post("/api/reindex", json={"sources": ["/etc"]})
        self.assertEqual(response.status_code, 400)

    def test_configuration_guard_requires_explicit_public_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit):
                demo_web.validate_demo_configuration()
        with patch.dict(os.environ, {"CITERAG_DEMO_PUBLIC": "1"}, clear=True):
            demo_web.validate_demo_configuration()

    def test_configuration_guard_rejects_production_auth_and_token_modes(self):
        for extra in (
            {"CITERAG_PRODUCTION": "1"},
            {"CITERAG_REQUIRE_AUTH": "1"},
            {"CITERAG_API_TOKEN": "never-print-me"},
        ):
            environment = {"CITERAG_DEMO_PUBLIC": "1"}
            environment.update(extra)
            with self.subTest(environment=environment):
                with patch.dict(os.environ, environment, clear=True):
                    with self.assertRaises(SystemExit):
                        demo_web.validate_demo_configuration()


if __name__ == "__main__":
    unittest.main()
