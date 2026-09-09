# Licenta-final

Aplicație dezvoltată pentru lucrarea de licență, dedicată prelucrării, integrării, anonimizării/pseudonimizării, analizei statistice și clusterizării nesupervizate a unui set de date clinice pulmonare.

Repository-ul conține codul sursă al aplicației. Setul de date medicale real, baza SQLite generată, modelele serializate și fișierele de output nu sunt incluse în repository.

## Funcționalități principale

- citirea și uniformizarea datelor din 12 foi Excel;
- anonimizarea/pseudonimizarea identificatorilor direcți;
- salvarea datelor prelucrate într-o bază SQLite locală;
- feature engineering și construirea matricilor pentru analiză;
- statistici descriptive și teste statistice;
- clustering KMeans și evaluarea mai multor valori pentru `k`;
- comparație KMeans, clustering ierarhic Ward și DBSCAN;
- metrici interne: Silhouette, Calinski-Harabasz și Davies-Bouldin;
- măsurători de runtime, throughput și scalabilitate;
- generarea automată de tabele, rapoarte și grafice;
- dashboard local realizat cu Streamlit;
- teste unitare și validarea output-urilor pipeline-ului.

## Structura principală

```text
.
├── build_features_v4_final_audit.py
├── train_ai_final.py
├── generate_descriptive_analysis.py
├── compare_clustering_methods.py
├── measure_engineering_performance.py
├── validate_pipeline_outputs.py
├── dashboard_app_engineering.py
├── run_pipeline_final.py
├── run_engineering_analysis.py
├── run_validation_final.py
├── run_all_with_dashboard.py
├── requirements.txt
├── requirements_testing.txt
├── requirements_specification.md
├── COD_AUDIT_CHECKLIST.md
└── tests/
    └── test_pipeline_unit.py
```

## Cerințe software

- Python 3.10 sau o versiune compatibilă mai nouă;
- `pip`;
- sistem de operare Windows, Linux sau macOS.

## Compilare

Aplicația este implementată în Python și nu necesită o etapă separată de compilare. Fișierele `.py` sunt executate direct de interpretul Python. Fișierele compilate automat de Python (`.pyc`, directoare `__pycache__`) sunt excluse din repository.

## Instalare

1. Clonați repository-ul și intrați în directorul proiectului:

```bash
git clone https://github.com/ioana1231/Licenta-final.git
cd Licenta-final
```

2. Creați un mediu virtual:

```bash
python -m venv .venv
```

3. Activați mediul virtual.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

4. Instalați dependențele:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Date de intrare

Setul de date real nu este publicat în repository deoarece conține date medicale și trebuie tratat conform cerințelor de confidențialitate.

Pentru rularea pipeline-ului, fișierul Excel de intrare trebuie plasat în directorul rădăcină al proiectului cu numele:

```text
date_intrare.xlsx
```

Scriptul `build_features_v4_final_audit.py` citește acest fișier și generează local:

```text
baza_date_licenta.db
```

Atât fișierul Excel, cât și baza de date generată sunt excluse din Git prin `.gitignore`.

## Lansarea aplicației

### Pipeline complet

Din directorul proiectului:

```bash
python run_pipeline_final.py
```

Comanda execută, în ordine:

1. prelucrarea și integrarea datelor;
2. generarea bazei SQLite și a caracteristicilor;
3. clusteringul KMeans final;
4. analiza statistică descriptivă;
5. comparația algoritmilor și măsurătorile inginerești;
6. testele unitare și validarea output-urilor.

### Pipeline complet + dashboard

```bash
python run_all_with_dashboard.py
```

La final este pornit dashboard-ul Streamlit pe localhost. Portul implicit este `8501`; dacă acesta este ocupat, scriptul caută automat un port disponibil apropiat.

Pentru generarea tuturor rezultatelor fără pornirea dashboard-ului:

```bash
python run_all_with_dashboard.py --no-dashboard
```

### Doar dashboard-ul

După ce output-urile au fost generate:

```bash
python -m streamlit run dashboard_app_engineering.py --server.address 127.0.0.1 --server.port 8501
```

Dashboard-ul rulează local și poate fi accesat din browser la `http://127.0.0.1:8501`.

## Output-uri generate local

În timpul rulării sunt create automat directoare precum:

```text
outputs_ai/
outputs_descriptive_analysis/
outputs_engineering/
reports/
```

Acestea conțin fișiere CSV, JSON, rapoarte, grafice și modele serializate. Sunt rezultate generate de aplicație și nu fac parte din codul sursă livrat în repository.

## Testare și validare

Validarea poate fi rulată separat cu:

```bash
python run_validation_final.py
```

sau doar testele unitare:

```bash
python -m pytest tests/test_pipeline_unit.py -q
```

## Confidențialitatea datelor

Repository-ul nu include setul de date medicale real și nici baza de date rezultată. Codul conține mecanisme pentru anonimizarea/pseudonimizarea identificatorilor direcți înainte de salvarea tabelelor utilizate ulterior în analiză.

Rezultatele de clustering au caracter exploratoriu și nu reprezintă o validare clinică.
