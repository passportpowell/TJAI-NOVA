#!/usr/bin/env python3
"""
Focused LLM API Key & Credential Scanner

This script specifically scans for:
1. LLM API keys (OpenAI, Claude, etc.)
2. Username/password combinations

It ignores Git objects and other binary files to reduce false positives
and provides specific details about what it finds.
"""

import os
import re
import sys
import argparse
import subprocess
from datetime import datetime
from typing import List, Dict, Tuple, Set, Optional, Any, Union

# ANSI color codes for terminal output
COLORS = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "white": "\033[97m",
    "bold": "\033[1m",
    "reset": "\033[0m"
}

# Focused regex patterns for LLM API keys and credentials
LLM_API_PATTERNS = {
    "openai_key": r"sk-[0-9a-zA-Z]{48}",  # OpenAI API key
    "claude_key": r"sk-ant-api[0-9a-zA-Z-]{24,}",  # Claude API key
    "groq_key": r"gsk_[0-9a-zA-Z]{32,}",  # Groq API key
    "perplexity_key": r"pplx-[0-9a-zA-Z]{32,}",  # Perplexity API key
    "cohere_key": r"[0-9a-zA-Z]{40}",  # Cohere API key
    "deepseek_key": r"sk-[0-9a-f]{32}",  # DeepSeek API key (different format from OpenAI)
    "gemini_key": r"AIza[0-9A-Za-z_\-]{35}",  # Google/Gemini API Key
    "mistral_key": r"[0-9a-zA-Z]{32,40}",  # Mistral API Key (general pattern)
}

# Username/password patterns
CREDENTIAL_PATTERNS = {
    "username_password": r"(?i)(username|user|login)\s*[=:]\s*['\"]?([a-zA-Z0-9_@.-]+)['\"]?",
    "password": r"(?i)(password|passwd|pass)\s*[=:]\s*['\"]?([a-zA-Z0-9_@#$%^&*()-+=<>,.?/\\|{}[\]~`!]+)['\"]?",
    "login_credential": r"(?i)(login|credential)\s*[=:]\s*['\"]?([a-zA-Z0-9_@.-]+)['\"]?",
    "plexos_credential": r"(?i)(PLEXOS_USERNAME|PLEXOS_PASSWORD)\s*[=:]\s*['\"]?([a-zA-Z0-9_@#$%^&*()-+=<>,.?/\\|{}[\]~`!]+)['\"]?"
}

# Files to ignore
IGNORE_PATTERNS = [
    r"\.git/",
    r"\.git\\",
    r"\.venv/",
    r"venv/",
    r"__pycache__/",
    r"node_modules/",
    # Binary files
    r"\.(jpg|jpeg|png|gif|svg|pdf|zip|tar|gz|mp4|mov|avi|ttf|woff)$",
    # Generated files
    r"\.(pyc|pyo|dll|so|exe)$"
]

class ApiKeyFinding:
    """Represents a found API key or credential"""
    def __init__(self, file_path: str, line_num: int, pattern_name: str, 
                 key_type: str, line_content: str, api_key: str):
        self.file_path = file_path
        self.line_num = line_num
        self.pattern_name = pattern_name
        self.key_type = key_type  # 'llm_api' or 'credential'
        self.line_content = line_content
        self.api_key = api_key
        
        # For display, mask the key except first and last few characters
        if len(api_key) > 10:
            self.masked_key = f"{api_key[:4]}...{api_key[-4:]}"
        else:
            self.masked_key = f"{api_key[:2]}...{api_key[-2:]}"
        
    def __str__(self) -> str:
        rel_path = os.path.relpath(self.file_path) if not self.file_path.startswith("git:") else self.file_path
        return f"{rel_path}:{self.line_num} [{self.pattern_name}] - {self.masked_key}"
    
    def detailed_str(self) -> str:
        """More detailed representation with line content"""
        rel_path = os.path.relpath(self.file_path) if not self.file_path.startswith("git:") else self.file_path
        return (f"File: {rel_path}\n"
                f"Line: {self.line_num}\n"
                f"Type: {self.key_type} ({self.pattern_name})\n"
                f"Key: {self.masked_key}\n"
                f"Content: {self.line_content.strip()}")

class ApiKeyScannerCLI:
    """Focused scanner for LLM API keys and credentials"""
    
    def __init__(self, repo_path: str = ".", verbose: bool = False):
        self.repo_path = os.path.abspath(repo_path)
        self.verbose = verbose
        self.findings: List[ApiKeyFinding] = []
        
    def _should_ignore_file(self, file_path: str) -> bool:
        """Check if a file should be ignored"""
        rel_path = os.path.relpath(file_path, self.repo_path)
        return any(re.search(pattern, rel_path) for pattern in IGNORE_PATTERNS)
    
    def scan_file(self, file_path: str) -> List[ApiKeyFinding]:
        """Scan a single file for API keys and credentials"""
        if self._should_ignore_file(file_path):
            return []
        
        findings = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f, 1):
                    # Check for LLM API keys
                    for pattern_name, pattern in LLM_API_PATTERNS.items():
                        for match in re.finditer(pattern, line):
                            api_key = match.group(0)
                            finding = ApiKeyFinding(
                                file_path=file_path,
                                line_num=i,
                                pattern_name=pattern_name,
                                key_type='llm_api',
                                line_content=line,
                                api_key=api_key
                            )
                            findings.append(finding)
                    
                    # Check for credentials
                    for pattern_name, pattern in CREDENTIAL_PATTERNS.items():
                        for match in re.finditer(pattern, line):
                            credential = match.group(0)  # Full match including username/password label
                            credential_value = match.group(2) if len(match.groups()) >= 2 else "unknown"
                            finding = ApiKeyFinding(
                                file_path=file_path,
                                line_num=i,
                                pattern_name=pattern_name,
                                key_type='credential',
                                line_content=line,
                                api_key=credential_value  # Just the credential value
                            )
                            findings.append(finding)
        except Exception as e:
            if self.verbose:
                print(f"Error scanning {file_path}: {str(e)}")
        
        return findings
    
    def scan_working_tree(self) -> List[ApiKeyFinding]:
        """Scan current files for API keys and credentials"""
        findings = []
        print(f"{COLORS['blue']}Scanning for LLM API keys and credentials...{COLORS['reset']}")
        
        # Create a list of files to scan
        files_to_scan = []
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                if not self._should_ignore_file(file_path):
                    files_to_scan.append(file_path)
        
        # Scan each file
        for i, file_path in enumerate(files_to_scan):
            if i % 50 == 0 or i == len(files_to_scan) - 1:
                print(f"Scanning file {i+1}/{len(files_to_scan)}...", end="\r")
            
            file_findings = self.scan_file(file_path)
            if file_findings:
                findings.extend(file_findings)
                rel_path = os.path.relpath(file_path, self.repo_path)
                
                # Group findings by type for better display
                llm_keys = [f for f in file_findings if f.key_type == 'llm_api']
                credentials = [f for f in file_findings if f.key_type == 'credential']
                
                if llm_keys:
                    print(f"{COLORS['red']}Found {len(llm_keys)} LLM API keys in {rel_path}{COLORS['reset']}")
                    for key in llm_keys:
                        print(f"  - Line {key.line_num}: {key.pattern_name} ({key.masked_key})")
                
                if credentials:
                    print(f"{COLORS['yellow']}Found {len(credentials)} credentials in {rel_path}{COLORS['reset']}")
                    for cred in credentials:
                        print(f"  - Line {cred.line_num}: {cred.pattern_name} ({cred.masked_key})")
        
        print(f"\n{COLORS['green']}Scan complete. Found {len(findings)} items.{COLORS['reset']}")
        return findings
    
    def scan_file_content(self, content: str, file_identifier: str) -> List[ApiKeyFinding]:
        """Scan a string content for API keys and credentials"""
        findings = []
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            # Check for LLM API keys
            for pattern_name, pattern in LLM_API_PATTERNS.items():
                for match in re.finditer(pattern, line):
                    api_key = match.group(0)
                    finding = ApiKeyFinding(
                        file_path=file_identifier,
                        line_num=i,
                        pattern_name=pattern_name,
                        key_type='llm_api',
                        line_content=line,
                        api_key=api_key
                    )
                    findings.append(finding)
            
            # Check for credentials
            for pattern_name, pattern in CREDENTIAL_PATTERNS.items():
                for match in re.finditer(pattern, line):
                    credential = match.group(0)
                    credential_value = match.group(2) if len(match.groups()) >= 2 else "unknown"
                    finding = ApiKeyFinding(
                        file_path=file_identifier,
                        line_num=i,
                        pattern_name=pattern_name,
                        key_type='credential',
                        line_content=line,
                        api_key=credential_value
                    )
                    findings.append(finding)
        
        return findings
    
    def scan_committed_files(self) -> List[ApiKeyFinding]:
        """Scan the most recent commit for API keys and credentials"""
        findings = []
        
        # Check if git is available and repo has commits
        try:
            # Get the most recent commit
            process = subprocess.Popen(
                ["git", "log", "-1", "--name-only", "--pretty=format:"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=self.repo_path
            )
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"{COLORS['yellow']}Could not retrieve recent commit: {stderr}{COLORS['reset']}")
                return findings
            
            # Get the list of files from the most recent commit
            recent_files = [line.strip() for line in stdout.splitlines() if line.strip()]
            
            print(f"{COLORS['blue']}Scanning {len(recent_files)} files from most recent commit...{COLORS['reset']}")
            
            for i, rel_file_path in enumerate(recent_files):
                file_path = os.path.join(self.repo_path, rel_file_path)
                
                # Skip if the file no longer exists or should be ignored
                if not os.path.exists(file_path) or self._should_ignore_file(file_path):
                    continue
                
                # Get the file content from the most recent commit
                process = subprocess.Popen(
                    ["git", "show", f"HEAD:{rel_file_path}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    cwd=self.repo_path
                )
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    if self.verbose:
                        print(f"Error retrieving {rel_file_path}: {stderr}")
                    continue
                
                # Scan the file content
                file_findings = self.scan_file_content(stdout, f"git:HEAD:{rel_file_path}")
                if file_findings:
                    findings.extend(file_findings)
                    
                    # Group findings by type for better display
                    llm_keys = [f for f in file_findings if f.key_type == 'llm_api']
                    credentials = [f for f in file_findings if f.key_type == 'credential']
                    
                    if llm_keys:
                        print(f"{COLORS['red']}Found {len(llm_keys)} LLM API keys in {rel_file_path}{COLORS['reset']}")
                        for key in llm_keys:
                            print(f"  - Line {key.line_num}: {key.pattern_name} ({key.masked_key})")
                    
                    if credentials:
                        print(f"{COLORS['yellow']}Found {len(credentials)} credentials in {rel_file_path}{COLORS['reset']}")
                        for cred in credentials:
                            print(f"  - Line {cred.line_num}: {cred.pattern_name} ({cred.masked_key})")
            
            print(f"\n{COLORS['green']}Commit scan complete. Found {len(findings)} items.{COLORS['reset']}")
            return findings
            
        except Exception as e:
            print(f"{COLORS['red']}Error scanning git history: {str(e)}{COLORS['reset']}")
            return findings
    
    def search_env_files(self) -> List[ApiKeyFinding]:
        """Specifically search for .env files and similar configuration"""
        findings = []
        env_patterns = [
            r"\.env$",
            r"\.env\.[a-zA-Z0-9]+$",  # .env.local, .env.production, etc.
            r".*config.*\.json$",
            r".*config.*\.js$",
            r".*config.*\.py$",
            r".*config.*\.toml$",
            r".*config.*\.yaml$",
            r".*config.*\.yml$",
            r".*secrets.*\.json$",
            r".*secrets.*\.py$",
            r".*credentials.*\.json$",
            r".*credentials.*\.py$"
        ]
        
        env_files = []
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.repo_path)
                
                # Skip .git directory
                if ".git" in rel_path.split(os.sep):
                    continue
                
                # Check if the file matches any env pattern
                if any(re.match(pattern, file) for pattern in env_patterns):
                    env_files.append(file_path)
        
        print(f"{COLORS['blue']}Scanning {len(env_files)} environment/config files...{COLORS['reset']}")
        
        for file_path in env_files:
            file_findings = self.scan_file(file_path)
            if file_findings:
                findings.extend(file_findings)
                rel_path = os.path.relpath(file_path, self.repo_path)
                
                # Group findings by type for better display
                llm_keys = [f for f in file_findings if f.key_type == 'llm_api']
                credentials = [f for f in file_findings if f.key_type == 'credential']
                
                if llm_keys:
                    print(f"{COLORS['red']}Found {len(llm_keys)} LLM API keys in {rel_path}{COLORS['reset']}")
                    for key in llm_keys:
                        print(f"  - Line {key.line_num}: {key.pattern_name} ({key.masked_key})")
                        # Show more context for API keys
                        print(f"    Context: {key.line_content.strip()}")
                
                if credentials:
                    print(f"{COLORS['yellow']}Found {len(credentials)} credentials in {rel_path}{COLORS['reset']}")
                    for cred in credentials:
                        print(f"  - Line {cred.line_num}: {cred.pattern_name} ({cred.masked_key})")
                        # Show more context for credentials
                        print(f"    Context: {cred.line_content.strip()}")
        
        return findings
    
    def run_scan(self, scan_env_only: bool = False, scan_committed: bool = False) -> List[ApiKeyFinding]:
        """Run a complete scan"""
        self.findings = []
        
        # Scan .env and config files first (highest priority)
        env_findings = self.search_env_files()
        self.findings.extend(env_findings)
        
        if not scan_env_only:
            # Scan working tree
            working_findings = self.scan_working_tree()
            self.findings.extend(working_findings)
            
            # Scan most recent commit if requested
            if scan_committed:
                commit_findings = self.scan_committed_files()
                self.findings.extend(commit_findings)
        
        return self.findings
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate a detailed report of findings"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.repo_path, f"api_key_scan_report_{timestamp}.md")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# LLM API Key & Credential Scan Report\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Repository: {self.repo_path}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            
            llm_keys = [f for f in self.findings if f.key_type == 'llm_api']
            credentials = [f for f in self.findings if f.key_type == 'credential']
            
            f.write(f"- **LLM API Keys found**: {len(llm_keys)}\n")
            f.write(f"- **Credentials found**: {len(credentials)}\n\n")
            
            # Group findings by file
            findings_by_file = {}
            for finding in self.findings:
                if finding.file_path not in findings_by_file:
                    findings_by_file[finding.file_path] = []
                findings_by_file[finding.file_path].append(finding)
            
            f.write(f"- **Files affected**: {len(findings_by_file)}\n\n")
            
            # Group by type
            if llm_keys:
                f.write("### LLM API Keys by Type\n\n")
                api_by_type = {}
                for finding in llm_keys:
                    if finding.pattern_name not in api_by_type:
                        api_by_type[finding.pattern_name] = []
                    api_by_type[finding.pattern_name].append(finding)
                
                for api_type, findings in api_by_type.items():
                    f.write(f"- **{api_type}**: {len(findings)}\n")
                f.write("\n")
            
            if credentials:
                f.write("### Credentials by Type\n\n")
                cred_by_type = {}
                for finding in credentials:
                    if finding.pattern_name not in cred_by_type:
                        cred_by_type[finding.pattern_name] = []
                    cred_by_type[finding.pattern_name].append(finding)
                
                for cred_type, findings in cred_by_type.items():
                    f.write(f"- **{cred_type}**: {len(findings)}\n")
                f.write("\n")
            
            # Detailed findings by file
            f.write("## Detailed Findings\n\n")
            
            for file_path, file_findings in findings_by_file.items():
                rel_path = os.path.relpath(file_path, self.repo_path) if not file_path.startswith("git:") else file_path
                f.write(f"### {rel_path}\n\n")
                
                # Group by type
                file_llm_keys = [f for f in file_findings if f.key_type == 'llm_api']
                file_credentials = [f for f in file_findings if f.key_type == 'credential']
                
                if file_llm_keys:
                    f.write("#### LLM API Keys\n\n")
                    for finding in file_llm_keys:
                        f.write(f"- **Line {finding.line_num}**: {finding.pattern_name}\n")
                        f.write(f"  - Key: `{finding.masked_key}`\n")
                        f.write(f"  - Context: `{finding.line_content.strip()}`\n\n")
                
                if file_credentials:
                    f.write("#### Credentials\n\n")
                    for finding in file_credentials:
                        f.write(f"- **Line {finding.line_num}**: {finding.pattern_name}\n")
                        f.write(f"  - Value: `{finding.masked_key}`\n")
                        f.write(f"  - Context: `{finding.line_content.strip()}`\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("1. **Remove all hardcoded API keys and credentials** from your codebase.\n")
            f.write("2. **Use environment variables** to store sensitive information.\n")
            f.write("3. **Update your .gitignore** to prevent committing sensitive files like .env.\n")
            f.write("4. **Revoke and rotate** any exposed credentials immediately.\n")
            f.write("5. **Consider using a secrets manager** for more secure credential handling.\n")
            
            # If credentials were found in Git history
            if any(f.file_path.startswith("git:") for f in self.findings):
                f.write("\n### Git History Cleaning\n\n")
                f.write("Credentials were found in your Git history. Consider using tools like `git filter-branch` ")
                f.write("or BFG Repo-Cleaner to remove sensitive data from your repository history.\n")
        
        print(f"{COLORS['green']}Report generated: {output_path}{COLORS['reset']}")
        return output_path

def main():
    parser = argparse.ArgumentParser(description="Focused LLM API Key & Credential Scanner")
    parser.add_argument("--repo", "-r", default=".", help="Path to repository")
    parser.add_argument("--output", "-o", help="Output report path")
    parser.add_argument("--env-only", action="store_true", help="Only scan .env and config files")
    parser.add_argument("--include-committed", action="store_true", help="Also scan most recent commit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    scanner = ApiKeyScannerCLI(repo_path=args.repo, verbose=args.verbose)
    
    # Run the scan
    scanner.run_scan(scan_env_only=args.env_only, scan_committed=args.include_committed)
    
    # Generate report
    if scanner.findings:
        scanner.generate_report(args.output)
    else:
        print(f"{COLORS['green']}No LLM API keys or credentials found.{COLORS['reset']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())