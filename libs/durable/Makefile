# Durable runtime: Rust kernel (crates/) and its Python package.
#
# PI_ROOT must point at a pi checkout with dependencies installed for the
# interop targets, which check SQLite files against pi-durable itself.

PI_NODE = node --conditions=source --experimental-strip-types interop/interop.ts
FIXTURES = crates/core/tests/fixtures

.PHONY: test interop pi-fixture

test:
	cargo nextest run -p durable-core

# Each implementation reads files written by the other exactly as their writer reads them.
interop:
	@test -n "$(PI_ROOT)" || (echo "PI_ROOT is not set" && exit 1)
	$(eval DIR := $(shell mktemp -d))
	PI_ROOT=$(PI_ROOT) $(PI_NODE) write $(DIR)/pi.sqlite
	PI_ROOT=$(PI_ROOT) $(PI_NODE) dump $(DIR)/pi.sqlite > $(DIR)/pi-by-pi.json
	cargo run -q --example interop -- dump $(DIR)/pi.sqlite > $(DIR)/pi-by-rust.json
	cmp $(DIR)/pi-by-pi.json $(DIR)/pi-by-rust.json
	cargo run -q --example interop -- write $(DIR)/rust.sqlite
	cargo run -q --example interop -- dump $(DIR)/rust.sqlite > $(DIR)/rust-by-rust.json
	PI_ROOT=$(PI_ROOT) $(PI_NODE) dump $(DIR)/rust.sqlite > $(DIR)/rust-by-pi.json
	cmp $(DIR)/rust-by-rust.json $(DIR)/rust-by-pi.json
	rm -r $(DIR)
	@echo "interop: both directions match"

# Rewrite the pi-written fixture the Rust tests read.
pi-fixture:
	@test -n "$(PI_ROOT)" || (echo "PI_ROOT is not set" && exit 1)
	rm -f $(FIXTURES)/pi-session.sqlite
	PI_ROOT=$(PI_ROOT) $(PI_NODE) write $(FIXTURES)/pi-session.sqlite
	PI_ROOT=$(PI_ROOT) $(PI_NODE) dump $(FIXTURES)/pi-session.sqlite > $(FIXTURES)/pi-session.json
