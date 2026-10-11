# Aşama 1 V14 — bağımsız doğrulama

- Kaynak: V13 yerel kopyası. Bu turda uygulama kodunda değişiklik yapılmadı.
- Tam test: `python -m pytest -q -p no:cacheprovider --durations=10`
- **113 passed in 30.08s**; sıfır başarısız test.
- En yavaş: `test_contrastive_encoder.py::test_full_scan_contrastive` 11.58 s.
- `tests/test_full_pipeline.py::test_learned_vs_random_representation_performance` 5.07 s.
- Ayrı full_pipeline: 4 passed in 19.97s.
- Tam `python -m src.run_chain_block_benchmarks` denemesi 200 saniye sınırına ulaştı, tamamlanmadı. Bu yüzden benchmark metrikleri bu turda yeniden doğrulanmadı.
- Tamamlanmamış benchmark PASS değildir. Güvenlik testlerinin geçmesi model performans hedeflerini doğrulamaz.
- GitHub deposuna yazma işlemi yapılmadı.
