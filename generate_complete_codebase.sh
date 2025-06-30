#!/bin/bash
OUTPUT="codebase.md"
echo "# ZeroVault AI Red Teaming Platform - Complete Codebase" > $OUTPUT
echo "" >> $OUTPUT
echo "Generated: $(date)" >> $OUTPUT
echo "" >> $OUTPUT
echo "## Project Structure" >> $OUTPUT
echo "" >> $OUTPUT
echo "\`\`\`" >> $OUTPUT
find . -type f \( -name "*.py" -o -name "*.js" -o -name "*.json" -o -name "*.md" -o -name "*.txt" -o -name "*.yml" -o -name "*.yaml" -o -name "*.sql" -o -name "*.sh" \) ! -path "./venv/*" ! -path "./__pycache__/*" ! -path "./.git/*" ! -path "./node_modules/*" ! -name "codebase.md" | sort >> $OUTPUT
echo "\`\`\`" >> $OUTPUT
echo "" >> $OUTPUT
echo "## Complete Source Code" >> $OUTPUT
echo "" >> $OUTPUT
find . -name "*.py" ! -path "./venv/*" ! -path "./__pycache__/*" ! -path "./.git/*" | sort | while read file; do
    echo "### $file" >> $OUTPUT
    echo "" >> $OUTPUT
    echo "\`\`\`python" >> $OUTPUT
    cat "$file" >> $OUTPUT
    echo "" >> $OUTPUT
    echo "\`\`\`" >> $OUTPUT
    echo "" >> $OUTPUT
done
for file in requirements.txt package.json .env.example Dockerfile docker-compose.yml database_schema.sql; do
    if [ -f "$file" ]; then
        echo "### $file" >> $OUTPUT
        echo "" >> $OUTPUT
        echo "\`\`\`" >> $OUTPUT
        cat "$file" >> $OUTPUT
        echo "" >> $OUTPUT
        echo "\`\`\`" >> $OUTPUT
        echo "" >> $OUTPUT
    fi
done
echo "Complete codebase documentation generated: $OUTPUT"
