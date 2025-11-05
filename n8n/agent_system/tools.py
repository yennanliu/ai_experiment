import datetime

def convert_time(time_str: str, target_timezone: str):
    """
    Converts time from one timezone to another.
    (This is a placeholder function)
    """
    return f"Time {time_str} converted to {target_timezone} is {datetime.datetime.now()}"

def http_request(url: str, method: str = "GET"):
    """
    Makes an HTTP request to a given URL.
    (This is a placeholder function)
    """
    return f"Made a {method} request to {url}. Response: 200 OK"

def financial_advice(query: str):
    """
    Provides financial advice.
    (This is a placeholder function and should not be used for real financial advice)
    """
    if "stock" in query:
        return "Investing in the stock market carries risks, but can also offer rewards."
    else:
        return "It's always a good idea to save for the future."

