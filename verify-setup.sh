#!/bin/bash

# Verification script for Next.js Portfolio Simulator setup

echo "======================================"
echo "Portfolio Simulator - Setup Verification"
echo "======================================"
echo ""

# Check Node.js version
echo "1. Checking Node.js installation..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node -v)
    echo "   ✓ Node.js installed: $NODE_VERSION"
    
    # Extract major version
    MAJOR_VERSION=$(echo $NODE_VERSION | cut -d'.' -f1 | sed 's/v//')
    if [ "$MAJOR_VERSION" -ge 18 ]; then
        echo "   ✓ Node.js version is 18 or higher"
    else
        echo "   ✗ Node.js version is below 18. Please upgrade."
        exit 1
    fi
else
    echo "   ✗ Node.js is not installed"
    echo "   Please install Node.js 18 or higher from https://nodejs.org/"
    exit 1
fi

echo ""

# Check npm
echo "2. Checking npm installation..."
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm -v)
    echo "   ✓ npm installed: v$NPM_VERSION"
else
    echo "   ✗ npm is not installed"
    exit 1
fi

echo ""

# Check directory structure
echo "3. Checking project structure..."
REQUIRED_DIRS=("app" "components" "lib")
REQUIRED_FILES=("package.json" "tsconfig.json" "next.config.js" "tailwind.config.js")

all_dirs_ok=true
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "   ✓ Directory exists: $dir/"
    else
        echo "   ✗ Missing directory: $dir/"
        all_dirs_ok=false
    fi
done

all_files_ok=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ File exists: $file"
    else
        echo "   ✗ Missing file: $file"
        all_files_ok=false
    fi
done

if [ "$all_dirs_ok" = false ] || [ "$all_files_ok" = false ]; then
    echo "   ✗ Project structure is incomplete"
    exit 1
fi

echo ""

# Check critical files
echo "4. Checking critical implementation files..."
CRITICAL_FILES=("lib/skewed-t.ts" "lib/simulation.ts" "components/SimulatorForm.tsx" "components/ResultsDisplay.tsx")

for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ Missing: $file"
        exit 1
    fi
done

echo ""

# Check if node_modules exists
echo "5. Checking dependencies..."
if [ -d "node_modules" ]; then
    echo "   ✓ node_modules exists"
    echo "   ✓ Dependencies appear to be installed"
else
    echo "   ⚠ node_modules not found"
    echo "   You need to run: npm install"
fi

echo ""
echo "======================================"
echo "Setup Verification Complete!"
echo "======================================"
echo ""

if [ -d "node_modules" ]; then
    echo "Ready to run! Execute:"
    echo "  npm run dev"
    echo ""
    echo "Then open: http://localhost:3000"
else
    echo "Next steps:"
    echo "  1. Run: npm install"
    echo "  2. Run: npm run dev"
    echo "  3. Open: http://localhost:3000"
fi

echo ""
echo "For more information, see:"
echo "  - README.md (comprehensive documentation)"
echo "  - QUICKSTART.md (quick start guide)"
echo "  - MIGRATION_NOTES.md (technical details)"
echo ""