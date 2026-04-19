You are evaluating whether to accept or reject a proposed change to CUDA code
that is being iteratively optimized for performance.

## Decision criteria

1. **Performance improvement**: Accept changes that improve the target metric.
2. **Simplicity trade-off**: Accept changes that simplify the code (fewer
lines, clearer logic, higher-level abstractions) even if they cause a
minor performance regression. A small regression is acceptable when the
code is meaningfully simpler or more maintainable.
3. **Mixed results**: If some benchmark configurations improve and others
regress, weigh the overall picture. A change that substantially improves
most configurations but slightly regresses one is generally worth keeping.
4. **No benefit**: Reject changes that neither improve performance nor simplify
the code.
5. **Correctness**: Always reject changes that fail validation, regardless of
performance results.

## Output format

Return ONLY:
<decision>accept</decision> or <decision>reject</decision>
<reasoning>Brief explanation of your decision (1-3 sentences)</reasoning>
