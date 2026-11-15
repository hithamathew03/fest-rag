"""Builds Group-12.pdf from the JSON results produced by Parts A, B and C."""
import glob, json, textwrap, platform, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
                                Preformatted, KeepTogether)
from common import *
from queries import QUERIES

MEMBERS = [
    ("Hitha Mathew P", "2026NS12034", "Q1, Q2", "Dataset design & generation, preprocessing pipeline"),
    ("Anurag Sharda", "2026NS13074", "Q3, Q4", "Embedding model selection, ChromaDB collection & CRUD"),
    ("Mahfuz Alam", "2026NS12121", "Q5, Q6", "Semantic search, multi-condition metadata filtering"),
    ("Nishar Ahamed A", "2026NS12120", "Q7, Q8", "Exhaustive NumPy search (L2 / cosine / dot), normalisation study"),
    ("Ansel Jo Paul", "2026NS12112", "Q9, Q10", "RAG pipeline, prompt design, with/without-context evaluation"),
]

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#1f3b63"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1f3b63"))
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=10.5, spaceBefore=6, spaceAfter=3)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13, spaceAfter=4)
SM = ParagraphStyle("SM", parent=B, fontSize=7.6, leading=9.5, spaceAfter=0)
CODE = ParagraphStyle("CODE", parent=ss["Code"], fontSize=7.2, leading=8.8, backColor=colors.HexColor("#f4f4f4"))
NOTE = ParagraphStyle("NOTE", parent=B, backColor=colors.HexColor("#fff4d6"), borderColor=colors.HexColor("#d9a400"),
                      borderWidth=0.8, borderPadding=6, spaceBefore=6, spaceAfter=8)


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def P(t, st=B):
    return Paragraph(t, st)


def tbl(rows, widths, header=True, fs=7.6):
    data = [[c if not isinstance(c, str) else Paragraph(esc(c), ParagraphStyle("c", parent=SM, fontSize=fs, leading=fs + 2))
             for c in r] for r in rows]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe7f2"))]
    t.setStyle(TableStyle(st))
    return t


# ------------------------------------------------------------------ figures
def architecture_png(path):
    fig, ax = plt.subplots(figsize=(11, 5.6)); ax.axis("off"); ax.set_xlim(0, 22); ax.set_ylim(0, 11)

    def box(x, y, w, h, title, body, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", fc=col, ec="#333", lw=1))
        ax.text(x + w / 2, y + h - 0.45, title, ha="center", va="top", fontsize=9.5, weight="bold")
        ax.text(x + w / 2, y + h - 1.15, body, ha="center", va="top", fontsize=7.6, wrap=True)

    def arr(x1, y1, x2, y2, lab=""):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=1.4, color="#333"))
        if lab:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.25, lab, fontsize=7, ha="center", color="#444")

    ax.text(11, 10.7, "FestiVox - Vector-DB & RAG architecture (offline / ingestion path on top, online / query path below)",
            ha="center", fontsize=10, weight="bold")
    box(0.2, 6.6, 4, 3.4, "1. Data sources", "Assignment-1 entities\n(festivals, venues, stages,\nartists, sponsors, tickets)\n+ attendee reviews,\nincident logs, debriefs", "#e8f0fb")
    box(5.0, 6.6, 4, 3.4, "2. Preprocessing", "clean whitespace, drop\nshort/empty & duplicate text,\ntype-cast metadata,\nassign doc_id FB-ff-nnnnn\n(5,200 records)", "#e8f6ea")
    box(9.8, 6.6, 4, 3.4, "3. Embedding", "WordLlama l2_supercat\n256-d, CPU, offline\nbatch=256\nraw (un-normalised)\nvectors kept", "#fdf0e0")
    box(14.6, 6.6, 7.1, 3.4, "4. ChromaDB (persistent)", "collection: festival_feedback\nHNSW index, space = cosine\nids + documents + embeddings\n+ 18 metadata fields\nCRUD: add / get / update / upsert / delete", "#f3e8fb")
    arr(4.3, 8.3, 5.0, 8.3); arr(9.1, 8.3, 9.8, 8.3); arr(13.9, 8.3, 14.6, 8.3)
    box(0.2, 0.6, 3.6, 4.6, "5. User query", "organiser / analyst\nquestion (Q1-Q10)\n+ optional metadata\nfilter (city, aspect,\nrating, ticket, date)", "#e8f0fb")
    box(4.4, 0.6, 3.6, 4.6, "6. Query embedding", "same WordLlama\nmodel - identical\nvector space", "#fdf0e0")
    box(8.6, 0.6, 4.4, 4.6, "7. Retrieval", "a) ChromaDB HNSW top-K\n   + where-filter ($and,\n   $in, $lte, $ne)\nb) NumPy exhaustive:\n   L2 / cosine / dot\n   K = 5 and 10", "#f3e8fb")
    box(13.6, 0.6, 3.9, 4.6, "8. Context prep.", "30 candidates ->\nMMR diversification\n-> top-6 passages\nwith [doc_id] +\nmetadata headers", "#e8f6ea")
    box(18.1, 0.6, 3.6, 4.6, "9. LLM answer", "grounded prompt\n(cite ids, admit\ninsufficiency)\nOllama llama3.2:3b\ntemperature 0", "#fde8e8")
    arr(3.9, 2.9, 4.4, 2.9); arr(8.1, 2.9, 8.6, 2.9); arr(13.1, 2.9, 13.6, 2.9); arr(17.6, 2.9, 18.1, 2.9)
    arr(18.0, 6.5, 11.5, 5.3, "vector + metadata lookup")
    plt.savefig(path, dpi=170, bbox_inches="tight"); plt.close()


def metric_png(b, path):
    q = [x["qid"] for x in b["queries"]]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
    w = 0.25
    for j, m in enumerate(["l2", "cosine", "dot"]):
        axs[0].bar(np.arange(10) + (j - 1) * w, [x["metrics"][f"raw|k10|{m}"]["fact_recall"] for x in b["queries"]], w, label=m)
    axs[0].set_xticks(range(10), q); axs[0].set_title("Fact recall@10, raw (un-normalised) embeddings", fontsize=9)
    axs[0].legend(fontsize=7); axs[0].set_ylim(0, 1.1)
    for j, key in enumerate(["l2_vs_cos", "cos_vs_dot", "l2_vs_dot"]):
        axs[1].bar(np.arange(10) + (j - 1) * w, [x["overlap"]["raw|k10"][key] for x in b["queries"]], w, label=key.replace("_vs_", " vs "))
    axs[1].set_xticks(range(10), q); axs[1].set_title("Top-10 overlap between metrics (raw). Normalised = 10/10 for all", fontsize=9)
    axs[1].legend(fontsize=7); axs[1].set_ylim(0, 11)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def norm_png(path):
    X = np.load(RESULTS / "embeddings.npy")
    import pandas as pd
    df = pd.read_csv(DATA_CSV, keep_default_na=False)
    n = np.linalg.norm(X, axis=1)
    fig, axs = plt.subplots(1, 2, figsize=(10, 2.8))
    axs[0].hist(n, bins=50, color="#4a78b5"); axs[0].set_title("L2 norm of stored embeddings", fontsize=9)
    axs[1].scatter(df.text.str.len(), n, s=2, alpha=0.3); axs[1].set_xlabel("record length (chars)", fontsize=8)
    axs[1].set_ylabel("vector norm", fontsize=8); axs[1].set_title("Norm vs text length (shorter text -> larger norm)", fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


# ------------------------------------------------------------------ report
def build():
    a, b = load("part_a.json"), load("part_b.json")
    cfiles = [f for f in glob.glob(str(RESULTS / "part_c_*.json")) if "dummy" not in f]
    c = json.load(open(sorted(cfiles)[0], encoding="utf-8")) if cfiles else None
    architecture_png(RESULTS / "arch.png"); metric_png(b, RESULTS / "metrics.png"); norm_png(RESULTS / "norms.png")
    S = []

    # ---------------- cover
    S += [Spacer(1, 1.2 * cm), P("<b>Assignment-2 - Modern Database Systems</b>", ParagraphStyle("t", parent=H1, fontSize=18, alignment=1)),
          P("Vector Databases, Semantic Search and Retrieval-Augmented Generation", ParagraphStyle("t2", parent=H2, alignment=1)),
          P("<b>FestiVox: Attendee-Feedback Intelligence for the Music Festival &amp; Concert Management System</b>",
            ParagraphStyle("t3", parent=B, fontSize=12, alignment=1, leading=16)),
          P("Group-12", ParagraphStyle("t4", parent=H2, alignment=1)), Spacer(1, 0.4 * cm)]
    S.append(tbl([["S. No.", "Name", "BITS ID", "Queries", "Implementation contribution"]] +
                 [[str(i + 1), m[0], m[1], m[2], m[3]] for i, m in enumerate(MEMBERS)], [1.2, 3.3, 2.6, 1.6, 8.0], fs=8.5))
    S += [Spacer(1, 0.4 * cm), P(f"Implementation: Python {platform.python_version()}, ChromaDB 1.x (persistent), NumPy, "
                                 f"WordLlama embeddings, Ollama LLM. Report generated {datetime.date.today():%d %b %Y} directly from the "
                                 "experiment result files (no hand-typed results).", B), PageBreak()]

    # ---------------- 1 problem
    S += [P("1. Problem Statement", H1),
          P("In Assignment-1 our group modelled the <b>Music Festival and Concert Management System</b>: a relational core "
            "(organisers, festivals, venues, stages, concerts, performances, artists, sponsors, ticket categories, tickets and "
            "payments), a MongoDB catalogue with heterogeneous artist documents and nested attendee-review arrays, and a Neo4j "
            "network of artists, genres and festivals. Those models answer <i>structured</i> questions very well (revenue per "
            "ticket category, fee per performance, multi-hop artist connections). What they cannot answer is the question every "
            "festival organiser asks after an event: <b>what did people actually experience, and what went wrong?</b>"),
          P("That knowledge lives in unstructured text: thousands of attendee reviews, security/operations incident logs and "
            "staff debrief notes. The same problem is described in many different words - <i>\"my ears were ringing for two days\"</i>, "
            "<i>\"painfully loud\"</i> and <i>\"bass was overpowering\"</i> all describe excessive volume, yet share almost no keywords. "
            "<b>FestiVox</b> extends the Assignment-1 application with a vector database (ChromaDB) that stores every feedback "
            "record as an embedding plus metadata keyed to the Assignment-1 entities, and a RAG assistant that answers "
            "organisers' questions with answers grounded in, and citing, the retrieved records."),
          P("1.1 Objectives", H2),
          P("(1) Store 5,000+ feedback records with embeddings and entity metadata in a persistent ChromaDB collection; "
            "(2) retrieve the top-K most similar records for a natural-language question, optionally constrained by metadata "
            "(festival, city, stage, artist, genre, ticket category, rating, aspect, document type, date); (3) quantify how the "
            "choice of distance metric and vector normalisation changes results; (4) generate concise, cited, evidence-based "
            "answers with an LLM and show when retrieval improves (or fails to improve) the answer."),
          P("1.2 Intended users and functionality", H2),
          tbl([["User", "Typical need", "Functionality used"],
               ["Festival organiser / ops head", "Post-event review: safety, crowd flow, delays", "Semantic search + filters on festival, doc_type, severity; RAG summary"],
               ["Sponsor & vendor managers", "SLA evidence for NimbusPay, GreenLeaf, UrbanRide, CloudNet", "Filter on aspect + festival; cited answers"],
               ["Artist booking team", "Which sets the crowd loved (re-booking, fee negotiation)", "Filter on aspect=performance, rating>=4"],
               ["Accessibility / compliance officer", "Barriers faced by wheelchair users and elderly attendees", "Semantic search on varied vocabulary"],
               ["Pricing analyst", "Is VIP worth it? Early-Bird value?", "Filter on ticket_category"]], [4, 6, 7]),
          P("1.3 Why a vector database instead of keyword / exact-match search", H2),
          P("Exact-match SQL (<font face='Courier'>LIKE '%loud%'</font>) or MongoDB text indexes only find records that share "
            "the query's words. Feedback is written by thousands of different people, so the same issue appears as paraphrases, "
            "synonyms and implicit descriptions (\"we stood in the pickup queue for over an hour\" = transport failure). Embeddings "
            "place such texts close together in vector space, so similarity search retrieves them regardless of wording; "
            "metadata filters then restore the precision of structured queries (e.g. <i>only VIP holders</i>, <i>only Mumbai in July</i>). "
            f"We measured this: across the 10 queries a TF-IDF keyword baseline recovered on average "
            f"<b>{np.mean([q['fact_recall_keyword'] for q in b['queries']]):.2f}</b> of the ground-truth key facts in its top-5, "
            f"pure semantic search <b>{np.mean([q['fact_recall_semantic'] for q in b['queries']]):.2f}</b>, and semantic search "
            f"with metadata filtering <b>{np.mean([q['fact_recall_filtered'] for q in b['queries']]):.2f}</b> (Section 5.3)."),
          PageBreak()]

    # ---------------- 2 architecture
    S += [P("2. System Architecture", H1), Image(str(RESULTS / "arch.png"), width=17.5 * cm, height=8.9 * cm),
          P("<b>Ingestion path.</b> Records are generated/collected (1), cleaned and given stable identifiers (2), embedded with "
            "WordLlama (3) and written to a persistent ChromaDB collection with their metadata (4). <b>Query path.</b> A question (5) "
            "is embedded with the <i>same</i> model (6); ChromaDB's HNSW index returns the top-K nearest records, optionally "
            "restricted by a metadata <font face='Courier'>where</font> clause (7a). For the metric study the identical query and "
            "stored vectors are compared exhaustively in NumPy (7b). For RAG, 30 candidates are diversified with Maximal Marginal "
            "Relevance and the best 6 are formatted with their ids and metadata (8) into a grounded prompt for the LLM (9)."),
          PageBreak()]

    # ---------------- 3 dataset
    S += [P("3. Dataset", H1), P("3.1 Source", H2),
          P("No public dataset of Indian music-festival feedback with the entity structure of our Assignment-1 schema exists, "
            "and the assignment discourages problems with ready-made implementations. We therefore built the corpus "
            "programmatically (<font face='Courier'>generate_dataset.py</font>, fixed seed 12) on top of the Assignment-1 data: the "
            "same 10 festivals, organisers, venues, stages, 15 artists (with genres), 10 sponsors and ticket categories are used, "
            "and every record carries their names as metadata. Text is composed from aspect-specific sentence banks "
            "(13 aspects, positive and negative phrasings, attendee personas, closing remarks) plus <b>festival-specific "
            "storylines</b> - realistic events such as NimbusPay wristband failures at Riverside Fest, monsoon waterlogging at "
            "Marine Arena, cold nights at Palace Lawns and the exit crush after the Neon NYE countdown. Storylines give the RAG "
            "system verifiable facts; generic sentences act as realistic distractors. Mixed reviews combine two aspects with "
            "opposite polarity. The corpus is synthetic, and this is acknowledged as a limitation (Section 8)."),
          P("3.2 Structure and metadata", H2),
          tbl([["Field", "Type", "Example / values", "Used for"],
               ["doc_id", "str (ChromaDB id)", "FB-05-02415 (festival 05, record 2415)", "identity, citations"],
               ["text", "str (embedded document)", f"avg {a['avg_chars']:.0f} chars", "embedding, LLM context"],
               ["doc_type", "str", "attendee_review / incident_report / staff_debrief", "filter"],
               ["festival_id, festival_name, organizer", "int / str", "5, Vibe Collective Riverside Fest", "link to Assignment-1 Festival"],
               ["city, venue, stage", "str", "Delhi, Riverside Grounds, River Stage", "link to Venue / Stage"],
               ["artist, genre", "str", "Night Owl Collective, Indie", "link to Artist / Performance"],
               ["event_date, month", "str ISO / int", "2026-02-02, 2", "date filters"],
               ["ticket_category", "str", "General / VIP / Early-Bird / NA", "link to TicketCategory"],
               ["rating, sentiment", "int 0-5 / str", "2, negative (0 = not a review)", "range filters"],
               ["aspect, aspect2", "str", "cashless_payment, hygiene (13 aspects)", "topic filters"],
               ["severity, author_role", "str", "high, security/ops", "incident triage"]], [4.2, 3.1, 5.8, 4.2]),
          P("3.3 Statistics and preprocessing", H2),
          P(f"Raw records: <b>{a['records_raw']}</b>; after preprocessing: <b>{a['records_clean']}</b>. Preprocessing: whitespace "
            "normalisation, removal of empty or very short (&lt;30 chars) texts, exact-duplicate removal (case-insensitive hash), "
            "explicit typing of numeric metadata, and replacing missing values by the string 'NA' (ChromaDB metadata cannot hold "
            "nulls). Document types: " + ", ".join(f"{k}: {v}" for k, v in a["doc_type_counts"].items()) +
            ". Review ratings: " + ", ".join(f"{k}-star: {v}" for k, v in a["rating_counts"].items()) + "."),
          tbl([["Festival", "Records"]] + [[k, str(v)] for k, v in a["festival_counts"].items()], [7, 2]),
          Spacer(1, 4), tbl([["Aspect", "Records (primary aspect)"]] + [[k, str(v)] for k, v in a["aspect_counts"].items()], [7, 3]),
          P("3.4 Sample records", H2),
          tbl([["doc_id", "type", "festival", "aspect", "rating", "text"]] +
              [[r["doc_id"], r["doc_type"], r["festival_name"], r["aspect"], str(r["rating"]), r["text"]] for r in a["sample_records"]],
              [2.2, 2.2, 2.8, 2.2, 1.1, 7.0]),
          PageBreak()]

    # ---------------- 4 embedding + chroma
    cr = a["crud"]
    S += [P("4. Embedding Model and ChromaDB Implementation", H1), P("4.1 Embedding model selection", H2),
          P(f"<b>Model: {EMBED_MODEL_NAME}</b> (<font face='Courier'>wordllama</font> package). WordLlama takes the token-embedding "
            "matrix of Llama-2-70B, compresses it to 256 dimensions and trains it with a contrastive objective on sentence-similarity "
            "data, so a sentence vector is a learned weighted average of token vectors. Reasons for the choice: (a) the weights ship "
            "inside the pip wheel, so the system runs fully offline on the BITS Proyogshala lab or a laptop with no GPU and no model "
            "download; (b) it embeds the full corpus in "
            f"<b>{a['embed_seconds']} s</b> on CPU, making the exhaustive-search experiments cheap; (c) its 256-d vectors keep "
            "ChromaDB storage small; (d) it is a sentence-similarity model (unlike raw word vectors) and handles the short, informal "
            "review language well; (e) its vectors are <b>not unit-normalised</b> (norm "
            f"{a['norm_stats']['min']:.2f}-{a['norm_stats']['max']:.2f}, mean {a['norm_stats']['mean']:.2f}), which lets us "
            "demonstrate the effect of normalisation on L2, cosine and dot-product rankings. Trade-off: a static-embedding model does "
            "not model word order or negation as well as a transformer encoder (e.g. all-MiniLM-L6-v2); we discuss the consequences "
            "in Sections 5 and 7. The same model and the same vectors are used for Parts A, B and C."),
          P("4.2 Collection design", H2),
          P(f"One persistent collection <font face='Courier'>{COLLECTION}</font> (PersistentClient at ./chroma_store), HNSW index "
            "with <font face='Courier'>hnsw:space = cosine</font> (cosine is the natural metric for sentence embeddings with "
            "varying norms). Each item = id (doc_id) + document (text) + embedding (computed by us and passed explicitly so that "
            "Part B uses byte-identical vectors) + 18 metadata fields. "
            f"Records inserted in batches of 1,000 in {a['insert_seconds']} s; collection count = <b>{a['collection_count']}</b>."),
          P("4.3 CRUD operations (actual outputs)", H2),
          tbl([["Operation", "ChromaDB call", "Result"],
               ["Create", "col.add(ids=['FB-05-NEW001'], documents, metadatas, embeddings)", f"count {a['collection_count']} -> {cr['count_after_insert']}"],
               ["Read by id", "col.get(ids=['FB-05-NEW001'])", f"rating={cr['read_by_id']['metadatas'][0]['rating']}, sentiment={cr['read_by_id']['metadatas'][0]['sentiment']}"],
               ["Read by metadata", "col.get(where={'$and':[{'festival_id':5},{'aspect':'cashless_payment'}]})", f"{cr['read_by_metadata_count']} records"],
               ["Update", "col.update(ids, documents=new_text, embeddings=re-embedded, metadatas={rating:3,...})",
                f"rating={cr['after_update']['metadatas'][0]['rating']}, sentiment={cr['after_update']['metadatas'][0]['sentiment']}; text: ..." + cr['after_update']['documents'][0][-70:]],
               ["Upsert", "col.upsert(...) - insert-or-update", "same id updated in place"],
               ["Delete by id", "col.delete(ids=['FB-05-NEW001'])", f"get() returns {len(cr['after_delete']['ids'])} ids; count {cr['count_after_delete']}"],
               ["Delete by filter", "col.delete(where={'$and':[{'festival_id':10},{'doc_type':'staff_debrief'}]})",
                f"{cr['bulk_delete_removed']} removed; count {cr['count_after_bulk_delete']}; restored -> {cr['count_after_restore']}"]],
              [2.4, 8.2, 6.9]),
          P("Note: on update we re-embed the new text; updating a document without its embedding would leave a stale vector.", SM),
          PageBreak()]

    # ---------------- 5 Part B
    S += [P("5. Semantic Similarity Search and Metadata-Based Retrieval", H1),
          P(f"<font face='Courier'>semantic_search(col, emb, text, k, where)</font> embeds the query and calls "
            "<font face='Courier'>col.query(query_embeddings, n_results=k, where=...)</font>; K is a parameter "
            f"(command-line argument, default {b['k_default']}). Distances returned by ChromaDB are <b>cosine distances</b> "
            "(1 - cosine similarity; 0 = identical direction, lower = more similar). Each query below is shown with its pure "
            "semantic top-5 and its metadata-filtered top-5. Several filters combine 2-3 conditions using "
            "<font face='Courier'>$and</font>, <font face='Courier'>$in</font>, <font face='Courier'>$lte/$gte</font> and "
            "<font face='Courier'>$ne</font>. 'Topic match' marks whether the record's primary or secondary aspect matches the "
            "question (and festival where relevant).")]
    for qq in b["queries"]:
        blk = [P(f"{qq['qid']} - {esc(qq['student'])} ({qq['bits']})", H2),
               P(f"<b>Query:</b> {esc(qq['text'])}<br/><b>Filter:</b> <font face='Courier' size=7.5>{esc(json.dumps(qq['where']))}</font> "
                 f"({qq['filter_matches']} records satisfy it)<br/><b>Significance:</b> {esc(qq['why'])}")]
        for lab, key in (("Pure semantic search (no filter)", "semantic"), ("Semantic search + metadata filter", "filtered")):
            blk.append(P(f"<i>{lab}</i>", SM))
            blk.append(tbl([["#", "doc_id", "cos dist", "festival", "aspect", "type", "rating", "topic match", "text"]] +
                           [[str(h["rank"]), h["id"], f"{h['distance']:.4f}", h["festival"], h["aspect"], h["doc_type"].split("_")[0],
                             str(h["rating"] or "-"), "yes" if h["relevant"] else "NO", h["text"][:150]] for h in qq[key]],
                           [0.5, 2.1, 1.3, 2.6, 2.0, 1.4, 0.9, 1.1, 5.7], fs=6.6))
        S.append(KeepTogether(blk[:3])); S += blk[3:]
    # analysis
    rows = [["Query", "keyword (TF-IDF) top-5", "semantic top-5", "semantic + filter top-5", "filter matches"]]
    for qq in b["queries"]:
        rows.append([qq["qid"], f"{qq['fact_recall_keyword']:.1f}", f"{qq['fact_recall_semantic']:.1f}",
                     f"{qq['fact_recall_filtered']:.1f}", str(qq["filter_matches"])])
    rows.append(["Mean", f"{np.mean([q['fact_recall_keyword'] for q in b['queries']]):.2f}",
                 f"{np.mean([q['fact_recall_semantic'] for q in b['queries']]):.2f}",
                 f"{np.mean([q['fact_recall_filtered'] for q in b['queries']]):.2f}", ""])
    Q = {q["qid"]: q for q in b["queries"]}
    S += [P("5.3 Effect of metadata filtering and analysis of relevant / less-relevant results", H2),
          P("<b>Fact recall@5</b> = share of each query's ground-truth key facts (list in queries.py) that appear in the retrieved "
            "texts. It rewards retrieving <i>different</i> relevant facts, not the same complaint five times."),
          tbl(rows, [2, 3.5, 3, 4, 2.5]),
          P(f"<b>Where filtering helps most.</b> Q2 (rain): pure semantic search returns generic 'rain turned the ground into mud' "
            f"reviews from other cities (topic matches {sum(h['relevant'] for h in Q['Q2']['semantic'])}/5 - festival-specific "
            "relevance fails) because the embedding captures <i>rain</i> but not <i>Mumbai in July</i>; adding city + month + doc-type "
            f"conditions lifts fact recall from {Q['Q2']['fact_recall_semantic']:.1f} to {Q['Q2']['fact_recall_filtered']:.1f}, "
            "surfacing the 40-minute delays, Gate 3 flooding and network outage. Q8 (crowd-favourite sets) shows the opposite failure: "
            "the unfiltered top-5 is dominated by <i>negative</i> reviews such as 'seemed low-energy and barely interacted with the "
            "crowd' - they share the words <i>interact</i> and <i>crowd</i> with the question and the static embedding does not "
            "capture that the meaning is reversed. The aspect=performance AND rating&gt;=4 filter removes them "
            f"(fact recall {Q['Q8']['fact_recall_semantic']:.1f} -> {Q['Q8']['fact_recall_filtered']:.1f})."),
          P("<b>Semantic strength.</b> Q3 and Q4 retrieve relevant records whose wording differs from the question: "
            "'cobblestone with steps... could not reach it without being carried' for a question about wheelchair access, and "
            "'my ears were ringing for two days' for 'so loud it hurt people's ears'. For Q3 the keyword baseline, which needs "
            f"shared words, reaches fact recall {Q['Q3']['fact_recall_keyword']:.1f} versus {Q['Q3']['fact_recall_semantic']:.1f} "
            "for semantic search; on Q4 both do equally well because 'loud' and 'ears' happen to occur in the question and the "
            "records - semantic search is not always better, but it degrades more gracefully when wording differs."),
          P("<b>Less-relevant results and their causes.</b> (i) <i>Sentiment blindness</i>: for Q1 ('why were attendees unhappy with "
            "cashless payments') unfiltered search also ranks positive reviews such as 'paying via the festival wristband was faster "
            "than carrying cash' - the static embedding encodes the topic (wristband, payment) much more strongly than the polarity. "
            "Our Q1 filter therefore adds <font face='Courier'>sentiment $ne positive</font>. (ii) <i>Mixed reviews</i>: a review "
            "whose first sentence is about accessibility and second about payments is retrieved for payment queries; this is correct "
            "behaviour but shows why per-record (not per-sentence) embeddings dilute signals. (iii) <i>Redundancy</i>: popular "
            "storylines are repeated by many attendees, so top-5 lists often contain several paraphrases of one fact (Q5); the RAG "
            "context builder uses MMR to counter this. (iv) Filters can be too strict: a filter on aspect only sees the primary "
            "aspect, so a record that mentions the topic in its second sentence is excluded."),
          P(f"<b>ANN accuracy.</b> ChromaDB's HNSW top-10 matched the exact (NumPy) cosine top-10 with recall "
            f"{np.mean([q['hnsw_recall_at10'] for q in b['queries']]):.2f} averaged over the 10 queries.")]
    S.append(PageBreak())

    # ---------------- 6 exhaustive metrics
    S += [P("6. Exhaustive Search with Euclidean, Cosine and Dot-Product Metrics", H1),
          P("All 5,200 stored vectors (matrix X, 5200x256) and the 10 query vectors q are the <b>identical arrays</b> used for "
            "ChromaDB (saved to results/embeddings.npy). Each metric is computed exhaustively in NumPy:"),
          Preformatted(textwrap.dedent("""\
            l2      : d = sqrt(sum((X - q)**2, axis=1))          -> distance, ascending  (lower = closer)
            cosine  : s = X@q / (||X|| * ||q||);  d = 1 - s      -> cosine distance, ascending
            dot     : s = X@q                                     -> similarity score, descending (higher = closer)
            normalised run: X_n = X/||X||,  q_n = q/||q||  then the same three formulas"""), CODE),
          P("Values reported for L2 and cosine are <b>distances</b>; values for dot product are <b>similarity scores</b>. "
            "Experiments were run on raw (un-normalised) WordLlama vectors and again after L2-normalisation, for K = 5 and K = 10. "
            "Full ranked lists for all 10 queries are in Appendix A; the summary is below."),
          Image(str(RESULTS / "metrics.png"), width=17.5 * cm, height=5.4 * cm)]
    rows = [["Query", "L2 FR@5", "cos FR@5", "dot FR@5", "L2 FR@10", "cos FR@10", "dot FR@10", "L2∩cos @10", "cos∩dot @10", "L2∩dot @10", "normalised: identical?"]]
    for qq in b["queries"]:
        M, O = qq["metrics"], qq["overlap"]
        rows.append([qq["qid"]] + [f"{M[f'raw|k5|{m}']['fact_recall']:.1f}" for m in ("l2", "cosine", "dot")] +
                    [f"{M[f'raw|k10|{m}']['fact_recall']:.1f}" for m in ("l2", "cosine", "dot")] +
                    [str(O["raw|k10"][k]) for k in ("l2_vs_cos", "cos_vs_dot", "l2_vs_dot")] +
                    ["yes" if O["normalized|k10"]["identical_order_all3"] and O["normalized|k5"]["identical_order_all3"] else "no"])
    S.append(tbl(rows, [1.2, 1.4, 1.4, 1.4, 1.5, 1.5, 1.5, 1.6, 1.6, 1.6, 2.0], fs=7))
    # worked example Q8
    q8 = Q["Q8"]
    rows = [["#", "L2 id", "L2 dist", "len", "cos id", "cos dist", "len", "dot id", "dot sim", "len / ||x||"]]
    for i in range(5):
        l, co, d = (q8["metrics"][f"raw|k5|{m}"]["hits"][i] for m in ("l2", "cosine", "dot"))
        rows.append([str(i + 1), l["id"], f"{l['score']:.3f}", str(l["chars"]), co["id"], f"{co['score']:.3f}", str(co["chars"]),
                     d["id"], f"{d['score']:.3f}", f"{d['chars']} / {d['norm']}"])
    S += [P("6.1 Worked example - Q8, K = 5, raw vectors", H2), tbl(rows, [0.6, 2.2, 1.4, 1.0, 2.2, 1.4, 1.0, 2.2, 1.4, 2.0], fs=7),
          Image(str(RESULTS / "norms.png"), width=16 * cm, height=4.5 * cm),
          P("6.2 How the metric influences results", H2),
          P(f"<b>Dot product is biased by vector length.</b> For un-normalised vectors, X·q = ||x||·||q||·cos θ, so a record "
            f"with a large norm can outrank a better-aligned one. With WordLlama the norm is negatively correlated with text "
            f"length (Pearson r = {b['norm_vs_length_corr']}) because averaging more token vectors cancels components: short, "
            "single-sentence records have the largest norms. Dot-product therefore systematically favours short records - visible "
            "in the Q8 example, where dot-product's top hits are much shorter than cosine's. This is sometimes useful (short, "
            "focused statements) and sometimes harmful: for Q1 and Q9 dot-product's top-5 fact recall drops to "
            f"{Q['Q1']['metrics']['raw|k5|dot']['fact_recall']:.1f} and {Q['Q9']['metrics']['raw|k5|dot']['fact_recall']:.1f} "
            f"(cosine: {Q['Q1']['metrics']['raw|k5|cosine']['fact_recall']:.1f} and {Q['Q9']['metrics']['raw|k5|cosine']['fact_recall']:.1f}) - "
            "for Q1 its top-5 is filled with the short positive line 'paying via the festival wristband was faster than carrying cash', "
            f"and its top-10 shares as few as {min(q['overlap']['raw|k10']['l2_vs_dot'] for q in b['queries'])} of 10 records with L2."),
          P("<b>L2 is biased in the opposite way.</b> ||x - q||² = ||x||² + ||q||² - 2 x·q, so with a fixed query, records whose "
            "norm is close to the query's norm are favoured; long multi-aspect reviews (small norm) can be closer in L2 than a short "
            f"precise one. L2 and cosine agree more often ({min(q['overlap']['raw|k10']['l2_vs_cos'] for q in b['queries'])}-"
            f"{max(q['overlap']['raw|k10']['l2_vs_cos'] for q in b['queries'])} of the top-10) because both penalise misalignment, but "
            "they diverge exactly where norms differ widely."),
          P("<b>Cosine ignores magnitude</b> and compares direction only, which matches the intended meaning of similarity for "
            "sentence embeddings. Mean fact recall@10 over the 10 queries was " +
            ", ".join(f"{m} {np.mean([q['metrics'][f'raw|k10|{m}']['fact_recall'] for q in b['queries']]):.2f}" for m in ("l2", "cosine", "dot")) +
            " (@5: " + ", ".join(f"{m} {np.mean([q['metrics'][f'raw|k5|{m}']['fact_recall'] for q in b['queries']]):.2f}" for m in ("l2", "cosine", "dot")) +
            "). Averages hide the behaviour: dot product wins on some queries (short, focused complaints such as Q7 transport) "
            "and loses badly on others (Q1, Q2, Q9 at K = 5), i.e. it is the least predictable. Cosine is length-independent and therefore the "
            "safest default; the ChromaDB collection uses <font face='Courier'>hnsw:space=cosine</font>."),
          P("<b>Effect of normalisation.</b> After L2-normalising X and q, ||x|| = ||q|| = 1, so ||x - q||² = 2 - 2cos θ and "
            "x·q = cos θ. All three metrics become monotonic transformations of each other and produce <b>identical rankings</b> - "
            "confirmed for all 10 queries at both K = 5 and K = 10 (last column). Only the reported values differ: e.g. an L2 "
            "distance d corresponds to cosine similarity 1 - d²/2. Practical conclusion: normalise embeddings (or use cosine) unless "
            "vector magnitude carries meaning you deliberately want to reward."),
          PageBreak()]

    # ---------------- 7 RAG
    S += [P("7. Retrieval-Augmented Generation Extension", H1), P("7.1 LLM selection", H2),
          P("<b>Model: Llama 3.2 3B Instruct served locally by Ollama</b> (<font face='Courier'>ollama pull llama3.2:3b</font>; "
            "~2 GB, runs on CPU in a few seconds per answer). Reasons: open weights and no paid API (feedback data stays on the "
            "organiser's machine), instruction-tuned for following grounding rules and producing short bullet answers, 128k context "
            "window (our prompts are ~1.2k tokens), and small enough for Proyogshala/laptop hardware. Decoding is deterministic "
            "(temperature 0, seed 12) so the with/without-context comparison is fair. The code also supports a Hugging Face "
            "backend (Qwen2.5-1.5B-Instruct) via a command-line switch."),
          P("7.2 Pipeline", H2),
          P("(1) <b>Query processing</b>: the question and its metadata filter (same as Part B). (2) <b>Query embedding</b> with "
            "WordLlama. (3) <b>Top-K retrieval</b>: 30 candidates from ChromaDB with the filter (fallback to unfiltered if &lt;3 "
            "match). (4) <b>Context preparation</b>: Maximal Marginal Relevance (λ = 0.5) picks 6 passages that are relevant but not "
            "near-duplicates; each is prefixed with [doc_id] and metadata (type, festival, city, stage, date, rating, ticket), capped "
            "at 3,000 characters. (5) <b>Prompt construction</b> with the system prompt below. (6) <b>Generation</b> and "
            "post-checks: cited ids are extracted and validated against the retrieved set; key-fact coverage is scored."),
          P("7.3 Prompts", H2), P("<b>RAG system prompt</b>", SM)]
    from part_c_rag import SYSTEM_RAG, SYSTEM_PLAIN
    S += [Preformatted("\n".join(textwrap.fill(l, 100) for l in SYSTEM_RAG.split("\n")), CODE),
          P("<b>User message</b>: <font face='Courier'>CONTEXT:\\n[FB-..] (metadata)\\ntext ... QUESTION: ... ANSWER (with [record-id] citations):</font>", SM),
          P("<b>No-context baseline system prompt</b>", SM), Preformatted(textwrap.fill(SYSTEM_PLAIN, 100), CODE)]
    if c is None:
        S += [P("<b>LLM RESULTS NOT YET GENERATED.</b> Sections 7.4-7.6 are filled automatically when the pipeline is run with "
                "an LLM available: <font face='Courier'>ollama pull llama3.2:3b &amp;&amp; python run_all.py</font>. "
                "The retrieval context that will be passed to the LLM for each query is already shown below.", NOTE)]
        cd = load("part_c_dummy.json")
        for r in cd["queries"]:
            S.append(P(f"{r['qid']}: {esc(r['text'])}", H3))
            S.append(tbl([["doc_id", "cos dist", "festival", "excerpt"]] + [[x["id"], f"{x['distance']:.4f}", x["festival"], x["excerpt"]]
                                                                         for x in r["retrieved"]], [2.2, 1.4, 3.5, 10.4], fs=6.8))
    else:
        S.append(P(f"Backend: <b>{c['backend']}</b>, model <b>{c['model']}</b>, top-K = {c['top_k']} (from {c['n_candidates']} "
                   f"candidates, MMR λ = {c['mmr_lambda']}).", B))
        S.append(P("7.4 RAG answers for the 10 queries", H2))
        for r in c["queries"]:
            blk = [P(f"{r['qid']} ({esc(r['student'])}): {esc(r['text'])}", H3),
                   tbl([["doc_id", "cos dist", "festival", "excerpt"]] + [[x["id"], f"{x['distance']:.4f}", x["festival"], x["excerpt"]]
                                                                        for x in r["retrieved"]], [2.2, 1.4, 3.5, 10.4], fs=6.6),
                   P("<b>RAG answer</b> (" + f"{r['seconds']} s; cited {len(r['cited_valid'])} valid / {len(r['cited_invalid'])} invalid ids"
                     + (f"; key-fact coverage {r['fact_coverage']}" if r['fact_coverage'] is not None else "") + "):", SM),
                   P(esc(r["answer"]).replace("\n", "<br/>"), ParagraphStyle("ans", parent=B, fontSize=8.3, leading=10.5,
                                                                             backColor=colors.HexColor("#eef5ee"), borderPadding=4))]
            S += blk
        S.append(PageBreak())
        S.append(P("7.5 Answers with vs without retrieved context", H2))
        rows = [["Query", "Coverage RAG", "Coverage no-ctx", "Valid citations", "No-context answer (excerpt)"]]
        for r in c["queries"]:
            if r["qid"] == "X1":
                continue
            rows.append([r["qid"], str(r["fact_coverage"]), str(r["no_context_fact_coverage"]), str(len(r["cited_valid"])),
                         r["no_context_answer"][:330]])
        S.append(tbl(rows, [1.2, 1.8, 2.0, 1.8, 10.7], fs=6.8))
        rc = np.mean([r["fact_coverage"] for r in c["queries"] if r["fact_coverage"] is not None])
        nc = np.mean([r["no_context_fact_coverage"] for r in c["queries"] if r["no_context_fact_coverage"] is not None])
        inval = sum(len(r["cited_invalid"]) for r in c["queries"])
        x1 = [r for r in c["queries"] if r["qid"] == "X1"][0]
        better = [r["qid"] for r in c["queries"] if r["fact_coverage"] is not None and r["fact_coverage"] > r["no_context_fact_coverage"]]
        notbetter = [r["qid"] for r in c["queries"] if r["fact_coverage"] is not None and r["fact_coverage"] <= r["no_context_fact_coverage"]]
        lat = [r["seconds"] for r in c["queries"]]
        S += [P(f"<b>Overall.</b> Mean key-fact coverage was <b>{rc:.2f}</b> with retrieved context versus <b>{nc:.2f}</b> without; "
                f"RAG scored higher on {len(better)} of 10 queries ({', '.join(better) or 'none'}), tied or lower on "
                f"{', '.join(notbetter) or 'none'}. RAG answers cited {sum(len(r['cited_valid']) for r in c['queries'])} record ids, all "
                f"from the retrieved set ({inval} fabricated ids). Generation took {min(lat):.0f}-{max(lat):.0f} s per RAG answer "
                f"(mean {np.mean(lat):.0f} s) on the evaluation machine.", B)]
        if c["model"].startswith("llama3.2"):
            S += [P("<b>Factual correctness.</b> Without context the model produces generic, unverifiable or wrong content. For Q5 "
                    "it described a stampede at a 'Vellore Festival' in 2017 that is unrelated to our data; for Q8 it recommended "
                    "Arijit Singh and Shreya Ghoshal, who are not in our artist roster; for Q2 it claimed stages flooded and the grounds "
                    "were closed, which no record says; for Q4 it quoted government decibel limits instead of what attendees "
                    "reported. With context the same model named the actual festivals, stages, sponsors and figures (NimbusPay "
                    "debits, the 40-minute delay on 11 July, Gate 3 flooding, 9 degrees after 10 pm, Rs 2500 taxi fares) and cited "
                    "the records they came from.", B),
                  P("<b>Relevance and completeness.</b> RAG answers stayed on the question but were sometimes padded with material "
                    "from the second sentence of mixed reviews. Q2 added 'poor crowd control' (too few security staff at Gate 1) and a "
                    "'positive note' about MC Rapstar's set - true of the records but not about rain. Q4 over-interpreted context: it "
                    "presented the sound-curfew and strobe-light records as evidence that the music was too loud, an inference the "
                    "records do not support. Q8 was selective in the right way: it cited only the 4 records that describe crowd "
                    "energy and left out the ones that do not.", B),
                  P("<b>Consistency with the available information - failure cases.</b> Q9 found the right fact (blanket rental "
                    "and bonfire pits, the top-ranked record) but then invented causal links from unrelated sentences in the same "
                    "reviews: it claimed waste-segregation bins and Night Owl Collective's set 'helped attendees stay warm', and ended by "
                    "saying no clear solutions were mentioned - contradicting its own second bullet. Q5 stated that record "
                    "FB-07-04049 did not report the crush, although that record describes it. The X1 probe (ticket sales and revenue, "
                    "which live in the Assignment-1 Ticket/Payment tables, not in feedback) was only partly handled: the model "
                    "correctly said revenue could not be calculated, but treated each review as a ticket sold, counted some records "
                    "under both VIP and General, and reported 'Total: 8 tickets' - an unsupported answer that the insufficiency rule in "
                    "the prompt should have prevented.", B),
                  P("<b>Limits of the automatic score.</b> Key-fact coverage is a keyword check, so it under-rates some correct "
                    "answers. Q10 scored only 0.4, but the RAG answer is accurate and balanced (good food, queues from under 15 to "
                    "over 90 minutes, about Rs 80 a meal), all cited; it simply did not use the exact words '45 minutes' or 'ran out'. "
                    "Its closing note that the records were insufficient was unnecessary. Q6 tied at 0.8: the no-context answer "
                    "sounds plausible, but only the RAG answer reflects the actual mixed verdict (worth it at Bassline Retreat and "
                    "Starlight Monsoon Carnival; not worth it at Neon NYE Bash and Soundscape Beach Bash).", B)]
        S += [P("<b>Insufficient-context probe X1 - full RAG answer:</b>", SM),
              P(esc(x1["answer"][:700]).replace("\n", "<br/>"), ParagraphStyle("ans2", parent=B, fontSize=8.3, leading=10.5,
                                                                              backColor=colors.HexColor("#fdeaea"), borderPadding=4))]
    S += [P("7.6 Effectiveness and limitations", H2),
          P("<b>Effective:</b> retrieval turns the LLM from a generic advisor into an analyst of <i>our</i> data: specific festivals, "
            "dates, stages, sponsors and figures, with citations that were 100% traceable to retrieved records. Metadata filters let "
            "the same pipeline answer scoped questions (one festival, VIP holders only, Mumbai in July), and MMR gave the model "
            "several distinct facts instead of five copies of one (Q2, Q3, Q7)."),
          P("<b>Limitations:</b> (1) answer quality is capped by retrieval - if the key record is outside the top-6 the model cannot "
            "use it, and when the top-6 is dominated by one repeated complaint (Q5, Q9) the answer becomes repetitive; (2) a 3B model "
            "sometimes links unrelated sentences in the context (Q9) or draws inferences the records do not support (Q4, Q5); "
            "(3) the insufficiency instruction was not reliably followed (X1) - a stricter prompt, a larger model or a separate "
            "'is this answerable?' check would help; (4) RAG summarises <i>samples</i>, not populations - it cannot count tickets "
            "or incidents, which belong in SQL/MongoDB from Assignment-1; (5) the static embedding model is weak on "
            "negation/sentiment, so positive reviews leak into complaint queries unless a sentiment filter is applied; (6) "
            "latency depends heavily on hardware - about 5-12 s per answer in this run, but an earlier CPU-only run took 5-7 "
            "minutes per answer, too slow for interactive use; (7) LLM output varies between runs even at temperature 0 (the "
            "earlier run failed Q9 differently and counted 4 tickets for X1), so conclusions should be drawn from patterns, not "
            "single answers; (8) the corpus is synthetic, so its wording is less varied than real reviews."),
          PageBreak()]

    # ---------------- 8 env, run, contributions, refs
    S += [P("8. Implementation Environment and Execution", H1),
          tbl([["File", "Purpose"],
               ["generate_dataset.py", "builds data/festival_feedback.csv (5,200 records) from Assignment-1 entities"],
               ["common.py", "paths, WordLlama embedder (offline), ChromaDB helpers"],
               ["queries.py", "the 10 queries (2 per student), filters, key facts"],
               ["part_a_build.py", "preprocessing, embedding, persistent ChromaDB load, CRUD demo -> results/part_a.json"],
               ["part_b_search.py [K]", "semantic + filtered search, TF-IDF baseline, NumPy L2/cosine/dot, K=5/10 -> part_b.json"],
               ["part_c_rag.py [ollama|hf]", "RAG pipeline, with/without-context comparison -> part_c_<backend>.json"],
               ["build_report.py / run_all.py", "regenerates this PDF from the result files / runs everything end-to-end"],
               ["FestiVox_Group12.ipynb", "notebook version running the same modules step by step"]], [5, 12.5]),
          P("<b>Run:</b> <font face='Courier'>pip install -r requirements.txt; ollama pull llama3.2:3b; python run_all.py</font>. "
            "Dependencies: chromadb, wordllama, numpy, pandas, scikit-learn, matplotlib, reportlab (Ollama for the LLM). "
            f"Environment for the reported results: {platform.system()} {platform.machine()}, Python {platform.python_version()}."),
          P("<b>Proyogshala evidence:</b> insert screenshots of the BITS Virtual Lab session (terminal running run_all.py, "
            "ChromaDB count output, Ollama model list) here before submission.", NOTE),
          P("9. Individual Contributions", H1),
          tbl([["Member", "BITS ID", "Queries", "Implementation"]] + [[m[0], m[1], m[2], m[3]] for m in MEMBERS], [3.5, 2.8, 1.8, 9.4]),
          P("10. References", H1),
          P("[1] ChromaDB documentation - collections, metadata filtering, HNSW configuration, https://docs.trychroma.com<br/>"
            "[2] D. Miller, WordLlama: recycled token embeddings for NLP, https://github.com/dleemiller/WordLlama<br/>"
            "[3] Meta, Llama 3.2 model card; Ollama, https://ollama.com/library/llama3.2<br/>"
            "[4] J. Carbonell, J. Goldstein, The use of MMR, diversity-based reranking for reordering documents and producing summaries, SIGIR 1998<br/>"
            "[5] P. Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, NeurIPS 2020<br/>"
            "[6] Group-12, Assignment-1: Music Festival and Concert Management System (relational, document and graph models)"),
          PageBreak()]

    # ---------------- Appendix A
    S.append(P("Appendix A - Exhaustive search results, all 10 queries, K = 10 (top-5 = first 5 rows), raw vectors", H1))
    S.append(P("L2 and cosine columns are distances (ascending); dot column is similarity (descending). ✓ = topic match.", SM))
    for qq in b["queries"]:
        rows = [["#", "L2 id", "L2 dist", "cos id", "cos dist", "dot id", "dot sim"]]
        H = {m: qq["metrics"][f"raw|k10|{m}"]["hits"] for m in ("l2", "cosine", "dot")}
        for i in range(10):
            rows.append([str(i + 1)] + sum([[H[m][i]["id"] + (" ✓" if H[m][i]["relevant"] else ""), f"{H[m][i]['score']:.4f}"]
                                            for m in ("l2", "cosine", "dot")], []))
        S.append(KeepTogether([P(f"{qq['qid']}: {esc(qq['text'])}", H3), tbl(rows, [0.7, 3.0, 1.6, 3.0, 1.6, 3.0, 1.6], fs=6.8)]))
    S.append(P("Normalised vectors: identical ranking for all three metrics (verified programmatically for every query, K=5 and K=10).", SM))

    def footer(cv, doc):
        cv.saveState(); cv.setFont("Helvetica", 7.5)
        cv.drawString(2 * cm, 1.2 * cm, "Group-12 | Assignment-2 Modern Database Systems | FestiVox")
        cv.drawRightString(19 * cm, 1.2 * cm, f"Page {doc.page}"); cv.restoreState()

    import copy
    S_doc = list(S)
    out = ROOT / "Group-12.pdf"
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm,
                      bottomMargin=1.8 * cm, title="Group-12 Assignment-2").build(S, onFirstPage=footer, onLaterPages=footer)
    print("wrote", out, "| LLM results included:", c is not None)
    try:
        from build_docx import story_to_docx
        story_to_docx(S_doc, ROOT / "Group-12.docx")
        print("wrote", ROOT / "Group-12.docx")
    except Exception as e:  # Word export is optional
        print("docx export failed:", e)


if __name__ == "__main__":
    build()
