"""Example: minimal RLM with mock LLM (no API key needed)."""

from __future__ import annotations

from rlm.repl import REPLEnvironment, parse_llm_response


def demo_repl_only():
    """Demonstrate REPL + parsing without any API calls."""

    context = """Date: Dec 12, 2022 || User: 63685 || Instance: How many years old is Benny Carter ?
Date: Dec 30, 2024 || User: 35875 || Instance: What war saw battles at Parrot's Beak ?
Date: Apr 13, 2024 || User: 67144 || Instance: What Metropolis landmark was first introduced ?
Date: Feb 29, 2024 || User: 67144 || Instance: When was Calypso music invented?
Date: Mar 15, 2024 || User: 53321 || Instance: What is the capital of France?
Date: May 10, 2024 || User: 67144 || Instance: Who painted the Mona Lisa?
Date: Jun 20, 2024 || User: 53321 || Instance: What is 2+2?
Date: Jul 04, 2024 || User: 38876 || Instance: What is the speed of light?"""

    repl = REPLEnvironment()
    repl.initialize(context)

    # Step 1: Peek at context
    print("=== Peek ===")
    output = repl.execute("context[:200]")
    print(f"First 200 chars:\n{output}")

    # Step 2: Find entries for specific user
    print("\n=== Grep for user 67144 ===")
    repl.execute(
        'lines_67144 = [l for l in context.split("\\n") if "User: 67144" in l]'
    )
    repl.execute("count_67144 = len(lines_67144)")
    repl.execute("print(count_67144)")
    output = repl.execute("count_67144")
    print(f"Entries for user 67144: {output}")

    # Step 3: Count all user groups
    print("\n=== Count users ===")
    repl.execute('all_lines = [l.strip() for l in context.split("\\n") if l.strip()]')
    repl.execute("users = set()")
    repl.execute("for l in all_lines:")
    repl.execute('    start = l.find("User: ")')
    repl.execute("    if start >= 0:")
    repl.execute("        user_id = l[start+6:start+12]")
    repl.execute("        users.add(user_id)")
    repl.execute("unique_count = len(users)")
    repl.execute("print(unique_count)")
    output = repl.execute("unique_count")
    print(f"Unique users: {output}")

    # Step 4: Simulate FINAL answer
    print("\n=== FINAL answer ===")
    sim_response = f"FINAL({output.strip()})"
    result = parse_llm_response(sim_response)
    print(f"Parsed: done={result.is_done}, answer={result.final_answer}")


if __name__ == "__main__":
    demo_repl_only()
