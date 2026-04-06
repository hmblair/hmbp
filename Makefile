CLAUDE_MD ?= $(HOME)/CLAUDE.md

.PHONY: install uninstall

install:
	@uv pip install --system -e .
	@if grep -q 'hmbp:start' $(CLAUDE_MD) 2>/dev/null; then \
		sed '/hmbp:start/,/hmbp:end/d' $(CLAUDE_MD) > $(CLAUDE_MD).tmp && \
		mv $(CLAUDE_MD).tmp $(CLAUDE_MD); \
	fi
	@sed 's|\./README\.md|$(CURDIR)/README.md|g' CLAUDE.md >> $(CLAUDE_MD)
	@echo 'Updated hmbp instructions in $(CLAUDE_MD)'

uninstall:
	@uv pip uninstall hmbp 2>/dev/null || true
	@if grep -q 'hmbp:start' $(CLAUDE_MD) 2>/dev/null; then \
		sed '/hmbp:start/,/hmbp:end/d' $(CLAUDE_MD) > $(CLAUDE_MD).tmp && \
		mv $(CLAUDE_MD).tmp $(CLAUDE_MD); \
		echo 'Removed hmbp instructions from $(CLAUDE_MD)'; \
	fi
