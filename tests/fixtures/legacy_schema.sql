-- Schema created by SQLModel.metadata.create_all at commit 600efa1 (before Alembic).
-- Regenerate: docker + 'git show 600efa1:src/domain/schemas.py' + create_all + pg_dump -s.
-- Used by tests/domain/test_migrations.py as the legacy starting point.

CREATE TYPE public.embeddingmodel AS ENUM (
    'CLAP',
    'MERT',
    'MUQ',
    'MUQ_MULAN',
    'ACOUSTIC_FEATURES'
);

CREATE TYPE public.status AS ENUM (
    'COMPLETED',
    'FAILED',
    'PENDING'
);

CREATE TYPE public.taskstatus AS ENUM (
    'PENDING',
    'RUNNING',
    'SUCCESS',
    'FAILURE',
    'SKIPPED'
);

CREATE TABLE public.song (
    created_at timestamp without time zone DEFAULT now() NOT NULL,
    updated_at timestamp without time zone DEFAULT now() NOT NULL,
    file_path character varying NOT NULL,
    title character varying,
    artist character varying,
    album character varying,
    genre character varying,
    duration_seconds integer,
    mtime timestamp without time zone
);

CREATE TABLE public.songembedding (
    created_at timestamp without time zone DEFAULT now() NOT NULL,
    updated_at timestamp without time zone DEFAULT now() NOT NULL,
    file_path character varying NOT NULL,
    model_name public.embeddingmodel NOT NULL,
    status public.status NOT NULL,
    dimension integer,
    milvus_collection_name character varying
);

CREATE TABLE public.songfeatures (
    created_at timestamp without time zone DEFAULT now() NOT NULL,
    updated_at timestamp without time zone DEFAULT now() NOT NULL,
    file_path character varying NOT NULL,
    status public.status NOT NULL,
    bpm double precision,
    spectral_centroid_mean double precision,
    spectral_centroid_std double precision,
    spectral_bandwidth_mean double precision,
    spectral_bandwidth_std double precision,
    mfcc_mean_0 double precision,
    mfcc_std_0 double precision,
    mfcc_mean_1 double precision,
    mfcc_std_1 double precision,
    mfcc_mean_2 double precision,
    mfcc_std_2 double precision,
    mfcc_mean_3 double precision,
    mfcc_std_3 double precision,
    mfcc_mean_4 double precision,
    mfcc_std_4 double precision,
    mfcc_mean_5 double precision,
    mfcc_std_5 double precision,
    mfcc_mean_6 double precision,
    mfcc_std_6 double precision,
    mfcc_mean_7 double precision,
    mfcc_std_7 double precision,
    mfcc_mean_8 double precision,
    mfcc_std_8 double precision,
    mfcc_mean_9 double precision,
    mfcc_std_9 double precision,
    mfcc_mean_10 double precision,
    mfcc_std_10 double precision,
    mfcc_mean_11 double precision,
    mfcc_std_11 double precision,
    mfcc_mean_12 double precision,
    mfcc_std_12 double precision,
    chroma_mean_0 double precision,
    chroma_std_0 double precision,
    chroma_mean_1 double precision,
    chroma_std_1 double precision,
    chroma_mean_2 double precision,
    chroma_std_2 double precision,
    chroma_mean_3 double precision,
    chroma_std_3 double precision,
    chroma_mean_4 double precision,
    chroma_std_4 double precision,
    chroma_mean_5 double precision,
    chroma_std_5 double precision,
    chroma_mean_6 double precision,
    chroma_std_6 double precision,
    chroma_mean_7 double precision,
    chroma_std_7 double precision,
    chroma_mean_8 double precision,
    chroma_std_8 double precision,
    chroma_mean_9 double precision,
    chroma_std_9 double precision,
    chroma_mean_10 double precision,
    chroma_std_10 double precision,
    chroma_mean_11 double precision,
    chroma_std_11 double precision
);

CREATE TABLE public.taskresult (
    created_at timestamp without time zone DEFAULT now() NOT NULL,
    updated_at timestamp without time zone DEFAULT now() NOT NULL,
    id uuid NOT NULL,
    name character varying NOT NULL,
    status public.taskstatus NOT NULL,
    progress integer NOT NULL,
    elapsed_time integer,
    message character varying,
    traceback character varying,
    details jsonb
);

ALTER TABLE ONLY public.song
    ADD CONSTRAINT song_pkey PRIMARY KEY (file_path);

ALTER TABLE ONLY public.songembedding
    ADD CONSTRAINT songembedding_pkey PRIMARY KEY (file_path, model_name);

ALTER TABLE ONLY public.songfeatures
    ADD CONSTRAINT songfeatures_pkey PRIMARY KEY (file_path);

ALTER TABLE ONLY public.taskresult
    ADD CONSTRAINT taskresult_pkey PRIMARY KEY (id);

CREATE INDEX ix_songfeatures_status ON public.songfeatures USING btree (status);

CREATE INDEX ix_taskresult_created_at ON public.taskresult USING btree (created_at);

CREATE INDEX ix_taskresult_name ON public.taskresult USING btree (name);

CREATE INDEX ix_taskresult_updated_at ON public.taskresult USING btree (updated_at);

ALTER TABLE ONLY public.songembedding
    ADD CONSTRAINT songembedding_file_path_fkey FOREIGN KEY (file_path) REFERENCES public.song(file_path);

ALTER TABLE ONLY public.songfeatures
    ADD CONSTRAINT songfeatures_file_path_fkey FOREIGN KEY (file_path) REFERENCES public.song(file_path);

