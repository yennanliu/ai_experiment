# Architectural Constraints

Every generated artifact must satisfy ALL of the following rules.
The constraint checker will flag violations and trigger a correction loop.

## 1. Security
The artifact must explicitly address authentication, authorization, or trust boundaries relevant to its component.
Signal words: "auth", "token", "permission", "trust", "secret", "credential", "TLS", "encrypt".

## 2. Interface Definition
Every component artifact must define its public interface: inputs, outputs, and any contracts or schemas.
Signal: an "Interface" or "API" section with typed parameters or a schema.

## 3. Error Handling
All known failure modes must be identified and a recovery or fallback strategy described.
Signal: an "Error Handling", "Failure Modes", or "Fallback" section.

## 4. Scalability
Performance characteristics must be stated: expected load, throughput limits, or horizontal scaling approach.
Signal words: "throughput", "RPS", "latency", "scale", "shard", "partition", "replica", "load".

## 5. Documentation
Each artifact must include a brief **Purpose** section (1–3 sentences) explaining why this component exists.
Signal: a "Purpose" or "Overview" header near the top.
