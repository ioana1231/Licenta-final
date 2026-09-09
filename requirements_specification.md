# Specificarea cerințelor pentru sistem

## Cerințe funcționale

| ID | Cerință | Implementare |
|---|---|---|
| F1 | Importul datelor din fișier Excel cu 12 foi | `build_features_v4_final_audit.py` citește foile definite în `SHEETS` |
| F2 | Curățarea și normalizarea datelor | Funcțiile `normalize_text`, `slugify_col`, `to_num`, `flag`, `flatten_columns` |
| F3 | Anonimizarea/pseudonimizarea pacienților | Mapare `cnp_key` → `id_pacient`; eliminarea identificatorilor direcți din tabelele salvate |
| F4 | Stocarea datelor în SQLite | Tabele `*_clean`, tabele de features și tabele de output salvate în `baza_date_licenta.db` |
| F5 | Construirea setului de caracteristici | `features_ai`, `features_clustering`, `features_clustering_descriptive` |
| F6 | Calculul scorurilor derivate | FACED/BSI parțiale și calculate complet, fără utilizare ca input în clustering |
| F7 | Clustering nesupervizat | KMeans final cu `k=3`, plus evaluarea valorilor k=2..8 |
| F8 | Compararea algoritmilor | KMeans vs hierarchical/Ward vs DBSCAN cu scoruri interne și runtime |
| F9 | Statistică descriptivă și validare exploratorie | Tabele/grafice pentru cohortă, simptome, comorbidități, etiologie, scoruri, teste statistice |
| F10 | Export rezultate | CSV, PNG, JSON, Markdown |
| F11 | Vizualizare locală | Dashboard Streamlit local, fără publicare online |
| F12 | Testare și validare | Unit tests, validare schemă DB, validare output-uri, sanity checks, stabilitate clustering |

## Cerințe nefuncționale

| ID | Cerință | Implementare / observație |
|---|---|---|
| NF1 | Securitatea datelor | Datele sunt anonimizate/pseudonimizate prin `id_pacient`; dashboard-ul rulează local |
| NF2 | Confidențialitate GDPR | Nu se publică dashboard-ul online; nu se includ identificatori direcți în tabelele de features/clustering |
| NF3 | Reproductibilitate | Scripturi rulate cap-coadă; `RANDOM_STATE`; rapoarte și metadate salvate |
| NF4 | Portabilitate | Python + SQLite + CSV/PNG; rulare locală din folderul proiectului |
| NF5 | Performanță | Runtime/throughput măsurate în `measure_engineering_performance.py` |
| NF6 | Scalabilitate | Test computațional prin replicarea matricei de clustering |
| NF7 | Observabilitate | Rapoarte Markdown/CSV pentru pașii principali și validare |
| NF8 | Interpretabilitate | Profiluri de cluster, top features, heatmap și tabele descriptive |

## Cazuri de utilizare

| Actor | Caz de utilizare | Rezultat |
|---|---|---|
| Student / cercetător | Rulează pipeline-ul de import și curățare | Bază SQLite și tabele curate |
| Student / cercetător | Generează caracteristici | `features_ai` și matrice de clustering |
| Student / cercetător | Rulează clustering | Etichete, metrici, profiluri, grafice |
| Student / cercetător | Compară algoritmi | Tabel KMeans vs hierarchical vs DBSCAN |
| Student / cercetător | Verifică performanța | Runtime, throughput, scalabilitate, refresh rate |
| Student / cercetător | Deschide dashboard local | Vizualizarea rezultatelor fără publicare online |
| Coordonator / evaluator | Inspectează rezultatele agregate | Tabele, grafice și structură SQLite, fără date identificabile |

## Observație metodologică

Sistemul are rol de prelucrare, analiză exploratorie și vizualizare locală. Rezultatele nu au rol diagnostic și necesită validare medicală interdisciplinară pentru interpretare clinică.
