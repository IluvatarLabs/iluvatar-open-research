# Contributing to IORI

Every contribution is welcome. You don't need to be a domain expert to participate.

## Five ways to contribute

### 1. Critique or challenge a finding

Open a [scientific critique](../../issues/new?template=scientific-critique.yml) issue. Push back on an interpretation, flag a confounder we missed, or challenge a confidence classification. We take every critique seriously: if it holds, we revise the finding; if it doesn't, we document why in the project's permanent record.

### 2. Contribute data

Open a [new data source](../../issues/new?template=new-data-source.yml) issue. Point us at a public dataset we missed, share curated or processed data, or contribute validation results. Include accession numbers or URLs so Marvin can ingest the data in the next iteration.

### 3. Suggest a new direction

Open a [literature suggestion](../../issues/new?template=literature-suggestion.yml) issue. Flag a paper we should have cited, propose a hypothesis Marvin should test, or suggest a method worth trying. Include the citation and explain why it matters for the project.

### 4. Validate a hypothesis

Open a [validation offer](../../issues/new?template=validation-offer.yml) issue. Tell us which hypothesis you want to test, what experimental or analytical system you'd use, and your estimated timeline. We'll coordinate to avoid duplicate effort and ensure you get full CRediT attribution on resulting publications.

### 5. Submit a new project idea

Open a [new project idea](../../issues/new?template=new-project-idea.yml) issue. Propose a question you think Marvin should run. Include:
- The research question (one sentence)
- Why it matters (2-3 sentences)
- What public data exists (bulleted list with accession numbers or URLs)
- What validation would look like

We reply to every submission with specific feedback, whether or not it's selected.

## How Marvin updates work

Marvin's iterations are delivered as pull requests against the relevant project directory. Each PR includes a structured summary: what changed, what datasets were added, which hypotheses were revised or retracted, and what open questions remain.

PRs sit open for **7 days** before merge so the community can review the diff, comment on specific changes, and flag concerns. After the review period, the PR merges and the project README is updated.

## Attribution

We use the [CRediT](https://credit.niso.org/) (Contributor Roles Taxonomy) framework:

- **Submitting researcher**: Conceptualization, Supervision, plus any roles actively filled
- **Iluvatar / Marvin team**: Methodology, Software, Formal Analysis, Data Curation, Visualization
- **Community contributors**: credited for specific roles based on actual contributions
- **Corresponding author**: submitting researcher (community projects) or Iluvatar lead scientist (showcase projects)
- **Marvin**: acknowledged in the methods section, not listed as an author

## Licensing

- Code, pipelines, environments: [Apache 2.0](LICENSE-CODE)
- Text, figures, manuscripts: [CC-BY 4.0](LICENSE-TEXT)
- Iluvatar takes no IP on any findings produced through this initiative

## Code of conduct

Be rigorous, be specific, be respectful. Critique methods and findings, not people. If you disagree with a conclusion, show the evidence.
