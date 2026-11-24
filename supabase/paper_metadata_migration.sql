-- Add extracted bibliographic metadata to existing paper records.
-- Safe to run more than once.
ALTER TABLE public.papers
  ADD COLUMN IF NOT EXISTS abstract text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS doi text,
  ADD COLUMN IF NOT EXISTS keywords text[] NOT NULL DEFAULT '{}',
  ADD COLUMN IF NOT EXISTS subject text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS pdf_metadata jsonb NOT NULL DEFAULT '{}'::jsonb;
