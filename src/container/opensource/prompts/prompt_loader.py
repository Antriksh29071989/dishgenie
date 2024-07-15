import logging

logger = logging.getLogger(__name__)


def get_prompt(context, query):
    """
    Provide prompt along with the context to feed in llm.
    :param context: Fetched from vector store.
    :param query: user query.
    :return: prompt with context and query.
    """
    logger.info("Adding context and user prompt in prompt.")
    base_prompt = """Based on the following context items, please answer the query.
    Give yourself room to think by extracting relevant passages from the context before answering the query.
    Don't return the thinking, only return the answer.
    Make sure your answers are as explanatory as possible.
    Use the following examples as reference for the ideal answer style.
    \nExample 1:
    Query: hat does the E15 error code mean on a LG dishwasher?
    Answer: The E15 error code on a LG dishwasher indicates that there is water in the base of the appliance, triggering
     the anti-flood device. This can be caused by a leak, a blocked drain hose, or a faulty water inlet valve. To resolve this, you may need to check for and remove any water in the base, inspect hoses and connections for leaks, and ensure the drain hose is not clogged or kinked.
    \nExample 2:
    Query: How do I clean the filter on my dishwasher?
    Answer: To clean the filter on your dishwasher, first, remove the bottom rack to access the filter assembly, 
    typically located at the bottom of the dishwasher. Twist and lift out the cylindrical filter and the flat filter 
    beneath it. Rinse both filters under warm running water to remove debris. For stubborn residue, use a soft brush or 
    cloth. Ensure all parts are thoroughly clean before reinstalling them in the dishwasher.
    \nExample 3:
    Query: What is the energy efficiency rating for dishwashers and why is it important?
    Answer: The energy efficiency rating for dishwashers is a measure of how effectively a dishwasher uses energy and 
    water. Ratings typically range from A+++ (most efficient) to D (least efficient) in Europe, and from 
    A (most efficient) to G (least efficient) in some other regions. An efficient dishwasher consumes less electricity 
    and water per cycle, which reduces utility bills and minimizes environmental impact. 
    Choosing a high-rated energy-efficient dishwasher helps conserve resources and saves money over the appliance's lifespan.
    \nRelevant passages: <extract relevant passages from the context here>
    User query: {query}
    Answer:"""
    return base_prompt.format(context=context,
                              query=query)
