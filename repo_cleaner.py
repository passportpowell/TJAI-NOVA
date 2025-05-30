#!/usr/bin/env python3
"""
Enhanced Git Repository Security Cleaner

This script scans your Git repository for sensitive information
and helps you clean it up automatically. It offers an interactive
menu-based interface for ease of use.

Run without arguments to enter interactive mode.
"""

import os
import re
import sys
import argparse
import subprocess
import shutil
import tempfile
import hashlib
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

# Regular expressions for sensitive data
REGEX_PATTERNS = {
    # API Keys
    "openai_key": r"sk-[0-9a-zA-Z]{48}",  # OpenAI API key
    "aws_access_key": r"AKIA[0-9A-Z]{16}",  # AWS Access Key ID
    "aws_secret_key": r"[0-9a-zA-Z/+]{40}",  # AWS Secret Access Key
    "google_api": r"AIza[0-9A-Za-z_\-]{35}",  # Google API Key
    "stripe_key": r"(pk|sk)_(test|live)_[0-9a-zA-Z]{24,}",  # Stripe API Key
    "sendgrid_key": r"SG\.[0-9a-zA-Z]{22}\.[0-9a-zA-Z]{43}",  # SendGrid API Key
    "jwt_token": r"eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}",  # JWT Token
    
    # Credentials
    "username_password": r"(?i)(username|user|login|password|passwd|secret|credential)[^=\n]{0,20}=[^\n]{3,}",
    "mongodb_uri": r"mongodb(\+srv)?://[^\s]+",
    "postgres_uri": r"postgresql://[^\s]+",
    "mysql_uri": r"mysql://[^\s]+",
    "redis_uri": r"redis://[^\s]+",
    
    # Environment variables
    "env_var": r"^[A-Z_]+=.+",
    
    # Generic API key/token patterns
    "generic_api_key": r"(?i)(api[_-]?key|apikey|api[_-]?token|access[_-]?token|secret[_-]?key)[^=\n]{0,20}=[^\n]{3,}"
}

# Files to ignore during scanning
IGNORE_PATTERNS = [
    r"\.git/",
    r"\.venv/",
    r"venv/",
    r"node_modules/",
    r"\.jpg$",
    r"\.jpeg$", 
    r"\.png$",
    r"\.gif$",
    r"\.svg$",
    r"\.pdf$",
    r"\.zip$",
    r"\.tar$",
    r"\.gz$",
    r"\.mp4$",
    r"\.mov$",
    r"\.avi$",
    r"\.ttf$",
    r"\.woff$"
]

# Combine all regex patterns
ALL_REGEX = "|".join(REGEX_PATTERNS.values())

class Finding:
    """Represents a sensitive data finding"""
    def __init__(self, file_path: str, line_num: int, pattern_name: str, 
                 pattern_type: str, content: str, value: str):
        self.file_path = file_path
        self.line_num = line_num
        self.pattern_name = pattern_name
        self.pattern_type = pattern_type
        self.content = content
        self.value = value
        self.hash = hashlib.md5(value.encode()).hexdigest()[:8]
        
    def __str__(self) -> str:
        return f"{self.file_path}:{self.line_num} [{self.pattern_name}] - {self.value[:10]}..."

class ConsoleMenu:
    """Simple console menu system"""
    
    @staticmethod
    def print_header(title):
        """Print a styled header"""
        width = min(os.get_terminal_size().columns, 80)
        print("\n" + "=" * width)
        print(f"{COLORS['bold']}{COLORS['cyan']}{title.center(width)}{COLORS['reset']}")
        print("=" * width + "\n")
    
    @staticmethod
    def print_options(options):
        """Print menu options"""
        for i, option in enumerate(options, 1):
            print(f"{COLORS['yellow']}{i}.{COLORS['reset']} {option}")
        # Add back/exit option
        print(f"{COLORS['yellow']}0.{COLORS['reset']} Back/Exit")
    
    @staticmethod
    def get_selection(options, prompt="Select an option", multi=False):
        """Get user selection from options"""
        if multi:
            prompt += " (comma-separated or 'all' for all options)"
        
        while True:
            try:
                user_input = input(f"{prompt}: ").strip()
                
                if user_input.lower() == 'all' and multi:
                    return list(range(1, len(options) + 1))
                
                if user_input == '0':
                    return [] if multi else 0
                
                if multi:
                    # Parse comma-separated values
                    selections = [int(x.strip()) for x in user_input.split(',')]
                    if all(1 <= s <= len(options) for s in selections):
                        return selections
                else:
                    # Single selection
                    selection = int(user_input)
                    if 0 <= selection <= len(options):
                        return selection
                
                print(f"{COLORS['red']}Invalid selection. Please try again.{COLORS['reset']}")
            except ValueError:
                print(f"{COLORS['red']}Please enter a number or comma-separated numbers.{COLORS['reset']}")
    
    @staticmethod
    def confirm(question="Are you sure?", default=False):
        """Ask for confirmation"""
        default_prompt = "Y/n" if default else "y/N"
        while True:
            response = input(f"{question} [{default_prompt}]: ").strip().lower()
            
            if not response:
                return default
            
            if response in ['y', 'yes']:
                return True
            if response in ['n', 'no']:
                return False
            
            print(f"{COLORS['red']}Please answer yes or no.{COLORS['reset']}")

class GitSecurityCleaner:
    """Main class for scanning and cleaning a Git repository"""
    
    def __init__(self, repo_path: str = ".", verbose: bool = False):
        self.repo_path = os.path.abspath(repo_path)
        self.verbose = verbose
        self.findings: List[Finding] = []
        self.backup_dir: Optional[str] = None
        self.has_git = self._check_git()
        
    def _check_git(self) -> bool:
        """Check if the directory is a git repository"""
        return os.path.isdir(os.path.join(self.repo_path, ".git"))
    
    def _run_command(self, command: List[str], cwd: Optional[str] = None) -> Tuple[str, str, int]:
        """Run a shell command and return stdout, stderr, and return code"""
        if cwd is None:
            cwd = self.repo_path
            
        if self.verbose:
            print(f"Running: {' '.join(command)}")
            
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, stderr = process.communicate()
        return stdout, stderr, process.returncode
    
    def _is_ignored(self, file_path: str) -> bool:
        """Check if a file should be ignored based on patterns"""
        rel_path = os.path.relpath(file_path, self.repo_path)
        return any(re.search(pattern, rel_path) for pattern in IGNORE_PATTERNS)
    
    def _backup_file(self, file_path: str) -> str:
        """Create a backup of a file before modifying it"""
        if not self.backup_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.backup_dir = os.path.join(tempfile.gettempdir(), f"git_cleaner_backup_{timestamp}")
            os.makedirs(self.backup_dir, exist_ok=True)
            
        rel_path = os.path.relpath(file_path, self.repo_path)
        backup_path = os.path.join(self.backup_dir, rel_path)
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def scan_file(self, file_path: str) -> List[Finding]:
        """Scan a single file for sensitive information"""
        file_findings = []
        
        if self._is_ignored(file_path):
            return file_findings
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f, 1):
                    for pattern_name, pattern in REGEX_PATTERNS.items():
                        for match in re.finditer(pattern, line):
                            pattern_type = "api_key" if "key" in pattern_name else "credential"
                            value = match.group(0)
                            
                            finding = Finding(
                                file_path=file_path,
                                line_num=i,
                                pattern_name=pattern_name,
                                pattern_type=pattern_type,
                                content=line.strip(),
                                value=value
                            )
                            file_findings.append(finding)
        except Exception as e:
            if self.verbose:
                print(f"Error scanning {file_path}: {str(e)}")
        
        return file_findings

    def scan_working_tree(self) -> List[Finding]:
        """Scan all files in the working tree"""
        findings = []
        
        print(f"{COLORS['blue']}Scanning working tree for sensitive information...{COLORS['reset']}")
        
        total_files = sum(len(files) for _, _, files in os.walk(self.repo_path))
        processed_files = 0
        
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                processed_files += 1
                if processed_files % 100 == 0 or processed_files == total_files:
                    print(f"Processed {processed_files}/{total_files} files...", end="\r")
                    
                file_path = os.path.join(root, file)
                if self._is_ignored(file_path):
                    continue
                    
                file_findings = self.scan_file(file_path)
                if file_findings:
                    findings.extend(file_findings)
                    rel_path = os.path.relpath(file_path, self.repo_path)
                    print(f"{COLORS['yellow']}Found {len(file_findings)} sensitive items in {rel_path}{COLORS['reset']}")
        
        print(f"\n{COLORS['green']}Working tree scan complete. Found {len(findings)} sensitive items.{COLORS['reset']}")
        return findings
    
    def scan_git_history(self) -> List[Finding]:
        """Scan git history for sensitive information"""
        if not self.has_git:
            print(f"{COLORS['red']}Not a git repository. Skipping history scan.{COLORS['reset']}")
            return []
            
        findings = []
        print(f"{COLORS['blue']}Scanning git history for sensitive information...{COLORS['reset']}")
        
        # Get all commit hashes
        stdout, _, _ = self._run_command(["git", "rev-list", "--all"])
        commit_hashes = stdout.splitlines()
        
        for i, commit_hash in enumerate(commit_hashes):
            if i % 10 == 0 or i == len(commit_hashes) - 1:  # Progress update every 10 commits
                print(f"Scanning commit {i+1}/{len(commit_hashes)}...", end="\r")
                
            # Get commit content
            stdout, _, _ = self._run_command(["git", "show", commit_hash])
            
            # Check for sensitive patterns
            for pattern_name, pattern in REGEX_PATTERNS.items():
                for match in re.finditer(pattern, stdout):
                    pattern_type = "api_key" if "key" in pattern_name else "credential"
                    value = match.group(0)
                    
                    finding = Finding(
                        file_path=f"git:commit:{commit_hash[:8]}",
                        line_num=0,  # Line number unknown in commit
                        pattern_name=pattern_name,
                        pattern_type=pattern_type,
                        content="[Git commit content]",
                        value=value
                    )
                    findings.append(finding)
                    
                    print(f"{COLORS['yellow']}Found sensitive {pattern_name} in commit {commit_hash[:8]}{COLORS['reset']}")
        
        print(f"\n{COLORS['green']}Git history scan complete. Found {len(findings)} sensitive items.{COLORS['reset']}")
        return findings
    
    def scan(self) -> List[Finding]:
        """Scan both working tree and git history"""
        self.findings = []
        
        # Scan working tree
        working_tree_findings = self.scan_working_tree()
        self.findings.extend(working_tree_findings)
        
        # Scan git history
        git_history_findings = self.scan_git_history()
        self.findings.extend(git_history_findings)
        
        return self.findings
    
    def clean_file(self, file_path: str, findings: List[Finding]) -> bool:
        """Clean sensitive data from a file"""
        if not os.path.exists(file_path) or file_path.startswith("git:"):
            return False
            
        # Create backup
        backup_path = self._backup_file(file_path)
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Replace sensitive data
            modified = False
            for finding in findings:
                if finding.file_path == file_path:
                    # Replace with a placeholder
                    placeholder = f"REMOVED_SENSITIVE_DATA_{finding.hash}"
                    new_content = content.replace(finding.value, placeholder)
                    if new_content != content:
                        content = new_content
                        modified = True
            
            # Write back if modified
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                print(f"{COLORS['green']}Cleaned sensitive data from {file_path}{COLORS['reset']}")
                return True
            
            return False
        except Exception as e:
            print(f"{COLORS['red']}Error cleaning {file_path}: {str(e)}{COLORS['reset']}")
            # Restore from backup
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, file_path)
            return False
    
    def clean_working_tree(self) -> int:
        """Clean sensitive data from current files"""
        if not self.findings:
            print(f"{COLORS['yellow']}No findings to clean.{COLORS['reset']}")
            return 0
            
        # Group findings by file
        files_to_clean = {}
        for finding in self.findings:
            if not finding.file_path.startswith("git:"):  # Skip git history findings
                if finding.file_path not in files_to_clean:
                    files_to_clean[finding.file_path] = []
                files_to_clean[finding.file_path].append(finding)
        
        # Clean each file
        cleaned_count = 0
        for file_path, file_findings in files_to_clean.items():
            if self.clean_file(file_path, file_findings):
                cleaned_count += 1
        
        return cleaned_count
    
    def clean_git_history(self) -> bool:
        """Clean sensitive data from git history"""
        if not self.has_git:
            print(f"{COLORS['red']}Not a git repository. Cannot clean history.{COLORS['reset']}")
            return False
            
        # Check for git filter-branch
        print(f"{COLORS['yellow']}Warning: Cleaning git history will rewrite commits.{COLORS['reset']}")
        print(f"{COLORS['yellow']}This is a destructive operation and should be used with caution.{COLORS['reset']}")
        
        if not ConsoleMenu.confirm("Are you sure you want to proceed with history cleaning?", False):
            print("History cleaning aborted.")
            return False
        
        # Extract unique sensitive values to replace
        sensitive_values = set()
        for finding in self.findings:
            sensitive_values.add(finding.value)
        
        if not sensitive_values:
            print(f"{COLORS['yellow']}No sensitive values found to clean from history.{COLORS['reset']}")
            return False
        
        # Create a filter-branch command to replace sensitive values
        filter_script = tempfile.NamedTemporaryFile(mode='w', delete=False)
        try:
            filter_script.write("#!/bin/sh\n")
            for value in sensitive_values:
                hash_value = hashlib.md5(value.encode()).hexdigest()[:8]
                placeholder = f"REMOVED_SENSITIVE_DATA_{hash_value}"
                # Use sed to replace sensitive data in all files
                filter_script.write(f"sed -i 's/{re.escape(value)}/{placeholder}/g' \"$@\"\n")
            filter_script.close()
            
            # Make script executable
            os.chmod(filter_script.name, 0o755)
            
            # Run git filter-branch
            print(f"{COLORS['blue']}Cleaning git history using filter-branch...{COLORS['reset']}")
            stdout, stderr, returncode = self._run_command([
                "git", "filter-branch", "--force", "--tree-filter",
                f"{filter_script.name} '$(git ls-files)'",
                "--", "--all"
            ])
            
            if returncode == 0:
                print(f"{COLORS['green']}Successfully cleaned git history.{COLORS['reset']}")
                print(f"{COLORS['yellow']}You may need to force push changes: git push --force{COLORS['reset']}")
                return True
            else:
                print(f"{COLORS['red']}Error cleaning git history:{COLORS['reset']}\n{stderr}")
                return False
                
        finally:
            os.unlink(filter_script.name)
    
    def update_gitignore(self) -> bool:
        """Add patterns to .gitignore for sensitive files"""
        gitignore_path = os.path.join(self.repo_path, ".gitignore")
        
        # Common patterns to ignore sensitive files
        patterns_to_add = [
            "# Added by security scanner",
            ".env",
            ".env.*",
            ".secrets/",
            "**/secrets.toml",
            "**/*_key*",
            "**/*password*",
            "**/credentials.*",
            "# API-specific files",
            "**/openai",
            "**/claude",
            "**/gemini",
            "**/groq",
            "**/deepseek",
            "**/perplexity"
        ]
        
        # Read existing .gitignore
        existing_patterns = []
        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                existing_patterns = [line.strip() for line in f.readlines()]
        
        # Find patterns to add (those not already in .gitignore)
        new_patterns = [p for p in patterns_to_add if p not in existing_patterns]
        
        if not new_patterns:
            print(f"{COLORS['blue']}No new patterns to add to .gitignore{COLORS['reset']}")
            return False
        
        # Add new patterns
        with open(gitignore_path, 'a', encoding='utf-8') as f:
            f.write("\n# Added by security scanner on " + datetime.now().strftime("%Y-%m-%d") + "\n")
            for pattern in new_patterns:
                f.write(pattern + "\n")
        
        print(f"{COLORS['green']}Added {len(new_patterns)} new patterns to .gitignore{COLORS['reset']}")
        return True
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate a report of findings"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.repo_path, f"security_scan_report_{timestamp}.txt")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Git Repository Security Scan Report\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Repository: {self.repo_path}\n\n")
            
            # Summary
            f.write("## Summary\n")
            f.write(f"Total findings: {len(self.findings)}\n")
            
            # Group by type
            findings_by_type = {}
            for finding in self.findings:
                if finding.pattern_type not in findings_by_type:
                    findings_by_type[finding.pattern_type] = []
                findings_by_type[finding.pattern_type].append(finding)
            
            for type_name, type_findings in findings_by_type.items():
                f.write(f"- {type_name}: {len(type_findings)}\n")
            
            # Details
            f.write("\n## Detailed Findings\n")
            for i, finding in enumerate(self.findings, 1):
                f.write(f"\n### Finding {i}\n")
                f.write(f"- File: {finding.file_path}\n")
                f.write(f"- Line: {finding.line_num}\n")
                f.write(f"- Type: {finding.pattern_type}\n")
                f.write(f"- Pattern: {finding.pattern_name}\n")
                f.write(f"- Value: {finding.value[:20]}...\n")
                
            f.write("\n## Recommendations\n")
            f.write("1. Review and clean all sensitive data from files\n")
            f.write("2. Consider using environment variables or a secrets manager\n")
            f.write("3. Update your .gitignore file to prevent committing sensitive files\n")
            f.write("4. Revoke and rotate any exposed credentials\n")
            f.write("5. Consider using git-secrets or pre-commit hooks to prevent future leaks\n")
        
        print(f"{COLORS['green']}Report generated: {output_path}{COLORS['reset']}")
        return output_path
    
    def interactive_menu(self) -> None:
        """Run the interactive menu system"""
        while True:
            ConsoleMenu.print_header("Git Repository Security Cleaner")
            
            if not self.findings:
                print("No scan has been performed yet. Please select 'Scan Repository' first.")
                scan_needed = True
            else:
                print(f"Repository: {self.repo_path}")
                print(f"Found {len(self.findings)} sensitive items.")
                scan_needed = False
            
            main_options = [
                "Scan Repository",
                "Clean Current Files",
                "Clean Git History (Advanced)",
                "Update .gitignore",
                "Generate Report",
                "Show Findings Summary",
                "Advanced Options"
            ]
            
            ConsoleMenu.print_options(main_options)
            choice = ConsoleMenu.get_selection(main_options, "Select an option")
            
            if choice == 0:  # Exit
                print("Exiting...")
                break
                
            elif choice == 1:  # Scan Repository
                self.scan()
                
            elif choice == 2:  # Clean Current Files
                if scan_needed:
                    print("Please scan the repository first.")
                    self.scan()
                
                if ConsoleMenu.confirm("Clean sensitive data from current files?"):
                    cleaned = self.clean_working_tree()
                    print(f"{COLORS['green']}Cleaned {cleaned} files.{COLORS['reset']}")
                
            elif choice == 3:  # Clean Git History
                if scan_needed:
                    print("Please scan the repository first.")
                    self.scan()
                
                self.clean_git_history()
                
            elif choice == 4:  # Update .gitignore
                self.update_gitignore()
                
            elif choice == 5:  # Generate Report
                if scan_needed:
                    print("Please scan the repository first.")
                    self.scan()
                
                output_path = input("Enter report output path (or leave empty for default): ").strip()
                if not output_path:
                    self.generate_report()
                else:
                    self.generate_report(output_path)
                
            elif choice == 6:  # Show Findings Summary
                if scan_needed:
                    print("Please scan the repository first.")
                    self.scan()
                
                self.show_findings_summary()
                
            elif choice == 7:  # Advanced Options
                self.show_advanced_options()
    
    def show_findings_summary(self) -> None:
        """Show a summary of findings"""
        if not self.findings:
            print(f"{COLORS['yellow']}No findings to display.{COLORS['reset']}")
            return
            
        ConsoleMenu.print_header("Findings Summary")
        
        # Group by type
        findings_by_type = {}
        for finding in self.findings:
            if finding.pattern_type not in findings_by_type:
                findings_by_type[finding.pattern_type] = []
            findings_by_type[finding.pattern_type].append(finding)
        
        for type_name, type_findings in findings_by_type.items():
            print(f"{COLORS['cyan']}{type_name.upper()}: {len(type_findings)} findings{COLORS['reset']}")
            
            # Group by pattern name
            findings_by_pattern = {}
            for finding in type_findings:
                if finding.pattern_name not in findings_by_pattern:
                    findings_by_pattern[finding.pattern_name] = []
                findings_by_pattern[finding.pattern_name].append(finding)
            
            for pattern_name, pattern_findings in findings_by_pattern.items():
                print(f"  {pattern_name}: {len(pattern_findings)} findings")
                
                # Show first few findings
                for i, finding in enumerate(pattern_findings[:3], 1):
                    rel_path = os.path.relpath(finding.file_path, self.repo_path) if not finding.file_path.startswith("git:") else finding.file_path
                    print(f"    {i}. {rel_path}:{finding.line_num}")
                
                if len(pattern_findings) > 3:
                    print(f"    ... and {len(pattern_findings) - 3} more")
        
        input("\nPress Enter to continue...")
    
    def show_advanced_options(self) -> None:
        """Show advanced options menu"""
        while True:
            ConsoleMenu.print_header("Advanced Options")
            
            advanced_options = [
                "Customize Ignore Patterns",
                "Search for Specific Pattern",
                "Clean Specific Files",
                "Set Verbose Mode",
                "View Backup Location"
            ]
            
            ConsoleMenu.print_options(advanced_options)
            choice = ConsoleMenu.get_selection(advanced_options, "Select an option")
            
            if choice == 0:  # Back
                break
                
            elif choice == 1:  # Customize Ignore Patterns
                self.customize_ignore_patterns()
                
            elif choice == 2:  # Search for Specific Pattern
                self.search_specific_pattern()
                
            elif choice == 3:  # Clean Specific Files
                self.clean_specific_files()
                
            elif choice == 4:  # Set Verbose Mode
                self.verbose = ConsoleMenu.confirm("Enable verbose mode?", self.verbose)
                print(f"Verbose mode: {'Enabled' if self.verbose else 'Disabled'}")
                
            elif choice == 5:  # View Backup Location
                if self.backup_dir:
                    print(f"Backup location: {self.backup_dir}")
                else:
                    print("No backups have been created yet.")
    
    def customize_ignore_patterns(self) -> None:
        """Customize ignore patterns"""
        ConsoleMenu.print_header("Customize Ignore Patterns")
        
        print("Current ignore patterns:")
        for i, pattern in enumerate(IGNORE_PATTERNS, 1):
            print(f"{i}. {pattern}")
        
        print("\nOptions:")
        print("1. Add pattern")
        print("2. Remove pattern")
        print("0. Back")
        
        choice = input("Select option: ").strip()
        
        if choice == "1":  # Add pattern
            new_pattern = input("Enter new pattern to ignore: ").strip()
            if new_pattern and new_pattern not in IGNORE_PATTERNS:
                IGNORE_PATTERNS.append(new_pattern)
                print(f"Added: {new_pattern}")
        
        elif choice == "2":  # Remove pattern
            try:
                index = int(input("Enter pattern number to remove: ").strip()) - 1
                if 0 <= index < len(IGNORE_PATTERNS):
                    removed = IGNORE_PATTERNS.pop(index)
                    print(f"Removed: {removed}")
                else:
                    print("Invalid pattern number.")
            except ValueError:
                print("Please enter a valid number.")
    
    def search_specific_pattern(self) -> None:
        """Search for a specific pattern"""
        ConsoleMenu.print_header("Search for Specific Pattern")
        
        pattern = input("Enter regex pattern to search for: ").strip()
        if not pattern:
            print("No pattern entered.")
            return
        
        try:
            re.compile(pattern)  # Validate regex
        except re.error:
            print(f"{COLORS['red']}Invalid regular expression.{COLORS['reset']}")
            return
        
        print(f"Searching for pattern: {pattern}")
        
        findings = []
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                if self._is_ignored(file_path):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for i, line in enumerate(f, 1):
                            for match in re.finditer(pattern, line):
                                value = match.group(0)
                                
                                finding = Finding(
                                    file_path=file_path,
                                    line_num=i,
                                    pattern_name="custom_pattern",
                                    pattern_type="custom",
                                    content=line.strip(),
                                    value=value
                                )
                                findings.append(finding)
                                
                                rel_path = os.path.relpath(file_path, self.repo_path)
                                print(f"Found match in {rel_path}:{i}")
                except Exception:
                    pass
        
        print(f"\nFound {len(findings)} matches.")
        
        if findings and ConsoleMenu.confirm("Add these findings to the main findings list?"):
            self.findings.extend(findings)
            print(f"Added {len(findings)} findings to the list.")
    
    def clean_specific_files(self) -> None:
        """Clean specific files only"""
        if not self.findings:
            print(f"{COLORS['yellow']}No findings to clean.{COLORS['reset']}")
            return
        
        ConsoleMenu.print_header("Clean Specific Files")
        
        # Group findings by file
        files_with_findings = {}
        for finding in self.findings:
            if not finding.file_path.startswith("git:"):  # Skip git history findings
                if finding.file_path not in files_with_findings:
                    files_with_findings[finding.file_path] = []
                files_with_findings[finding.file_path].append(finding)
        
        # Create a list of files with sensitive data
        files = list(files_with_findings.keys())
        if not files:
            print("No files with sensitive data found in current working tree.")
            return
        
        print("Files with sensitive data:")
        for i, file_path in enumerate(files, 1):
            rel_path = os.path.relpath(file_path, self.repo_path)
            print(f"{i}. {rel_path} ({len(files_with_findings[file_path])} findings)")
        
        # Get selection
        selections = ConsoleMenu.get_selection(files, "Select files to clean", multi=True)
        if not selections:
            print("No files selected.")
            return
        
        # Clean selected files
        cleaned_count = 0
        for index in selections:
            file_path = files[index - 1]
            if self.clean_file(file_path, files_with_findings[file_path]):
                cleaned_count += 1
        
        print(f"{COLORS['green']}Cleaned {cleaned_count} files.{COLORS['reset']}")

def main():
    # Check if running directly without arguments
    if len(sys.argv) == 1:
        # Run interactive mode
        cleaner = GitSecurityCleaner()
        cleaner.interactive_menu()
        return 0
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Git Repository Security Cleaner")
    parser.add_argument("action", choices=["scan", "clean"], nargs="?", default="interactive", 
                      help="Action to perform (default: interactive)")
    parser.add_argument("--repo", "-r", default=".", help="Path to repository")
    parser.add_argument("--output", "-o", help="Output report path")
    parser.add_argument("--auto", action="store_true", help="Automatic cleaning without prompts")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    cleaner = GitSecurityCleaner(repo_path=args.repo, verbose=args.verbose)
    
    if args.action == "interactive" or not args.action:
        # Run interactive mode
        cleaner.interactive_menu()
    elif args.action == "scan":
        # Only scan
        cleaner.scan()
        if args.output:
            cleaner.generate_report(args.output)
        else:
            cleaner.generate_report()
    elif args.action == "clean":
        # Scan and clean
        cleaner.scan()
        if args.auto:
            # Automatic cleaning
            cleaner.clean_working_tree()
            cleaner.update_gitignore()
            if args.output:
                cleaner.generate_report(args.output)
        else:
            # Interactive cleaning
            cleaner.interactive_menu()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())