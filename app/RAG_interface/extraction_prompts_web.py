"""
Extraction prompts for web data extraction
"""

prompts_to_run = {
    "part_number": {
        "web_prompt": """
Extract the part number from the website data. Look for patterns like:
- Numbers with hyphens (e.g., 2330171-2)
- Numbers with letters (e.g., 1-2330171-2)
- Numbers with special characters (e.g., 2330171-2.000)

If multiple part numbers are found, return the most prominent one (usually the one in the title or header).
If no part number is found, return "NOT FOUND".
"""
    },
    "description": {
        "web_prompt": """
Extract the product description or title from the website data. Look for:
- Product name or title
- Short description
- Product type or category

If multiple descriptions are found, return the most detailed one.
If no description is found, return "NOT FOUND".
"""
    },
    "manufacturer": {
        "web_prompt": """
Extract the manufacturer name from the website data. Look for:
- Company name
- Brand name
- "Manufactured by" or similar phrases

If multiple manufacturer names are found, return the most prominent one.
If no manufacturer is found, return "NOT FOUND".
"""
    },
    "material": {
        "web_prompt": """
Extract the material information from the website data. Look for:
- Material type
- Material composition
- Material specifications

If multiple materials are found, return all of them separated by commas.
If no material information is found, return "NOT FOUND".
"""
    },
    "dimensions": {
        "web_prompt": """
Extract the dimensions from the website data. Look for:
- Length, width, height measurements
- Diameter
- Unit of measurement (mm, inch, etc.)

Format the dimensions as "L x W x H" with units.
If no dimensions are found, return "NOT FOUND".
"""
    },
    "voltage_rating": {
        "web_prompt": """
Extract the voltage rating from the website data. Look for:
- Voltage specifications
- Voltage range
- Maximum/minimum voltage
- Unit (V, kV, etc.)

If multiple voltage ratings are found, return all of them separated by commas.
If no voltage rating is found, return "NOT FOUND".
"""
    },
    "current_rating": {
        "web_prompt": """
Extract the current rating from the website data. Look for:
- Current specifications
- Current range
- Maximum/minimum current
- Unit (A, mA, etc.)

If multiple current ratings are found, return all of them separated by commas.
If no current rating is found, return "NOT FOUND".
"""
    },
    "temperature_rating": {
        "web_prompt": """
Extract the temperature rating from the website data. Look for:
- Operating temperature range
- Maximum/minimum temperature
- Unit (°C, °F, etc.)

If multiple temperature ratings are found, return all of them separated by commas.
If no temperature rating is found, return "NOT FOUND".
"""
    },
    "contact_resistance": {
        "web_prompt": """
Extract the contact resistance from the website data. Look for:
- Contact resistance value
- Maximum contact resistance
- Unit (mΩ, Ω, etc.)

If multiple contact resistance values are found, return all of them separated by commas.
If no contact resistance is found, return "NOT FOUND".
"""
    },
    "insulation_resistance": {
        "web_prompt": """
Extract the insulation resistance from the website data. Look for:
- Insulation resistance value
- Minimum insulation resistance
- Unit (MΩ, GΩ, etc.)

If multiple insulation resistance values are found, return all of them separated by commas.
If no insulation resistance is found, return "NOT FOUND".
"""
    }
} 