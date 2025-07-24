import streamlit as st
import ast
import re
import os
from groq import Groq


# Initialize Groq client
@st.cache_resource
def get_groq_client():
    """
    Initialize and cache the Groq client.
    Assumes GROQ_API_KEY is set in environment variables or Streamlit secrets.
    """
    try:
        # Try to get API key from Streamlit secrets first, then environment
        api_key = "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"
        if not api_key:
            st.error("GROQ_API_KEY not found. Please set it in your environment variables or Streamlit secrets.")
            return None
        
        client = Groq(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {str(e)}")
        return None

def analyze_function_with_ast(code_string: str) -> dict:
    """
    Analyze Python function using AST to extract metadata.
    
    Args:
        code_string (str): The Python function code to analyze
        
    Returns:
        dict: Function metadata including name, args, return hints, etc.
    """
    try:
        tree = ast.parse(code_string)
        func_node = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
        
        if not func_node:
            return None
        
        # Extract function details
        func_info = {
            'name': func_node.name,
            'args': [],
            'return_annotation': None,
            'has_decorators': len(func_node.decorator_list) > 0,
            'is_async': isinstance(func_node, ast.AsyncFunctionDef)
        }
        
        # Extract arguments with type hints
        for arg in func_node.args.args:
            arg_info = {'name': arg.arg}
            if arg.annotation:
                arg_info['type_hint'] = ast.unparse(arg.annotation)
            func_info['args'].append(arg_info)
        
        # Extract return type annotation
        if func_node.returns:
            func_info['return_annotation'] = ast.unparse(func_node.returns)
            
        return func_info
        
    except SyntaxError as e:
        st.error(f"Syntax error in code: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error analyzing code: {str(e)}")
        return None

def create_structured_prompt(code_string: str, func_info: dict) -> str:
    """
    Create a well-structured prompt for the AI model.
    
    Args:
        code_string (str): The original function code
        func_info (dict): Function metadata from AST analysis
        
    Returns:
        str: Structured prompt for docstring generation
    """
    args_list = [arg['name'] for arg in func_info['args']]
    type_hints = {arg['name']: arg.get('type_hint', 'Unknown') for arg in func_info['args']}
    
    prompt = f"""You are an expert developer. Generate a comprehensive Google-style docstring for the following function.

REQUIREMENTS:
1. Write a clear, concise one-line summary
2. Include detailed Args section with type information and explanations
3. Include Returns section with type and description
4. Include Raises section if applicable
5. Add Examples section if the function is complex
6. Use proper Google docstring format

FUNCTION INFORMATION:
- Function name: {func_info['name']}
- Arguments: {', '.join(args_list)}
- Return type hint: {func_info.get('return_annotation', 'Not specified')}
- Is async: {func_info.get('is_async', False)}

FUNCTION CODE:
```python
{code_string}
```

Generate ONLY the docstring content (without triple quotes). The docstring should be professional, accurate, and follow Google style guidelines exactly."""

    return prompt

def generate_docstring_with_groq(code_string: str, model_name: str = "llama-3.1-70b-versatile") -> str:
    """
    Generate docstring using Groq API with a powerful model.
    
    Args:
        code_string (str): The Python function code
        model_name (str): Groq model to use (default: llama-3.1-70b-versatile)
        
    Returns:
        str: Generated docstring or error message
    """
    client = get_groq_client()
    if not client:
        return "Error: Could not initialize Groq client"
    
    # Analyze the function
    func_info = analyze_function_with_ast(code_string)
    if not func_info:
        # Fallback for non-function code
        prompt = f"""Generate a brief comment explaining what this Python code does:

```python
{code_string}
```

Provide only a concise explanation without code formatting."""
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            comment = response.choices[0].message.content.strip()
            return f"# {comment}\n{code_string}"
        except Exception as e:
            return f"Error generating comment: {str(e)}"
    
    # Create structured prompt
    prompt = create_structured_prompt(code_string, func_info)
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert developer who writes perfect Google-style docstrings. Always follow the exact format requested and be concise but comprehensive."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.2,  # Lower temperature for more consistent output
            top_p=0.9
        )
        
        docstring = response.choices[0].message.content.strip()
        
        # Clean up the docstring and format it properly
        docstring = clean_docstring(docstring)
        
        # Insert docstring into function
        return insert_docstring_into_function(code_string, docstring)
        
    except Exception as e:
        st.error(f"Error calling Groq API: {str(e)}")
        return code_string

def clean_docstring(docstring: str) -> str:
    """
    Clean and format the generated docstring.
    
    Args:
        docstring (str): Raw docstring from AI
        
    Returns:
        str: Cleaned docstring
    """
    # Remove any markdown code blocks
    docstring = re.sub(r'```python\n?|```\n?', '', docstring)
    
    # Remove any triple quotes that might have been included
    docstring = re.sub(r'"""', '', docstring)
    
    # Ensure proper indentation
    lines = docstring.split('\n')
    cleaned_lines = []
    for line in lines:
        if line.strip():
            cleaned_lines.append('    ' + line.strip())
        else:
            cleaned_lines.append('')
    
    return '\n'.join(cleaned_lines)

def insert_docstring_into_function(code_string: str, docstring: str) -> str:
    """
    Insert the docstring into the function at the correct position.
    
    Args:
        code_string (str): Original function code
        docstring (str): Generated docstring
        
    Returns:
        str: Function code with docstring inserted
    """
    lines = code_string.split('\n')
    
    # Find the function definition line
    func_def_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('def ') and ':' in line:
            func_def_idx = i
            break
    
    if func_def_idx == -1:
        return code_string
    
    # Insert docstring after function definition
    result_lines = lines[:func_def_idx + 1]
    result_lines.append('    """')
    result_lines.extend(docstring.split('\n'))
    result_lines.append('    """')
    result_lines.extend(lines[func_def_idx + 1:])
    
    return '\n'.join(result_lines)

def main():
    """
    Main Streamlit application for docstring generation.
    """
    st.title("🤖 AI-Powered Docstring Generator")
    st.write("Generate high-quality Google-style docstrings using Groq's powerful language models.")
    
    # Model selection
    model_options = [
        "deepseek-r1-distill-llama-70b",
        "llama-3.1-8b-instant", 
        "meta-llama/llama-guard-4-12b",
        "gemma2-9b-it"
    ]
    
    selected_model = st.selectbox(
        "Choose AI Model:",
        model_options,
        help="Different models offer various trade-offs between speed and quality"
    )
    
    # Code input
    st.subheader("📝 Enter Your Function")
    code_input = st.text_area(
        "Paste your Python function here:",
        height=200,
        placeholder="""def example_function(param1, param2):
    # Your function code here
    return result"""
    )
    
    # Generate button
    if st.button("🚀 Generate Docstring", type="primary"):
        if not code_input.strip():
            st.warning("Please enter some Python code.")
            return
            
        with st.spinner(f"Generating docstring using {selected_model}..."):
            result = generate_docstring_with_groq(code_input, selected_model)
        
        st.subheader("✨ Generated Result")
        st.code(result, language="python")
        
        # Copy button
        st.download_button(
            label="📋 Download Code",
            data=result,
            file_name="function_with_docstring.py",
            mime="text/plain"
        )

    # Instructions
    with st.expander("📚 Usage Instructions"):
        st.markdown("""
        ### How to use:
        1. **Set up your Groq API key**: Add `GROQ_API_KEY` to your environment variables or Streamlit secrets
        2. **Paste your Python function** in the text area above
        3. **Select an AI model** (llama-3.1-70b-versatile recommended for best quality)
        4. **Click "Generate Docstring"** to get your enhanced function
        
        ### Supported features:
        - ✅ Google-style docstring format
        - ✅ Automatic argument analysis
        - ✅ Type hint detection
        - ✅ Return value documentation
        - ✅ Error handling documentation
        - ✅ Multiple powerful AI models
        
        ### API Key Setup:
        ```bash
        # Option 1: Environment variable
        export GROQ_API_KEY="your_api_key_here"
        
        # Option 2: Streamlit secrets
        # Create .streamlit/secrets.toml:
        GROQ_API_KEY = "your_api_key_here"
        ```
        """)

if __name__ == "__main__":
    main()