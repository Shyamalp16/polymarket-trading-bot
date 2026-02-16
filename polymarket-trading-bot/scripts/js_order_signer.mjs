#!/usr/bin/env node

import readline from "node:readline";
import { Wallet } from "ethers";

const privateKey = process.env.POLY_SIGNER_PRIVATE_KEY || "";
const chainId = Number(process.env.POLY_SIGNER_CHAIN_ID || "137");

if (!privateKey) {
  process.stderr.write("POLY_SIGNER_PRIVATE_KEY is required\n");
  process.exit(2);
}

const wallet = new Wallet(privateKey);

const CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E";
const NEG_RISK_CTF_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a";

const orderTypes = {
  Order: [
    { name: "salt", type: "uint256" },
    { name: "maker", type: "address" },
    { name: "signer", type: "address" },
    { name: "taker", type: "address" },
    { name: "tokenId", type: "uint256" },
    { name: "makerAmount", type: "uint256" },
    { name: "takerAmount", type: "uint256" },
    { name: "expiration", type: "uint256" },
    { name: "nonce", type: "uint256" },
    { name: "feeRateBps", type: "uint256" },
    { name: "side", type: "uint8" },
    { name: "signatureType", type: "uint8" },
  ],
};

function getDomain(negRisk) {
  return {
    name: "Polymarket CTF Exchange",
    version: "1",
    chainId,
    verifyingContract: negRisk ? NEG_RISK_CTF_EXCHANGE : CTF_EXCHANGE,
  };
}

async function handleRequest(req) {
  const id = req?.id ?? null;
  const method = req?.method ?? "";
  if (method === "health") {
    return { id, ok: true, address: wallet.address };
  }
  if (method === "sign_order") {
    const params = req?.params || {};
    const negRisk = Boolean(params.negRisk);
    const message = params.message || {};
    const signature = await wallet.signTypedData(getDomain(negRisk), orderTypes, message);
    return { id, signature };
  }
  return { id, error: `Unsupported method: ${method}` };
}

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

rl.on("line", async (line) => {
  let req;
  try {
    req = JSON.parse(line);
  } catch (err) {
    process.stdout.write(JSON.stringify({ id: null, error: `Invalid JSON: ${String(err)}` }) + "\n");
    return;
  }

  try {
    const res = await handleRequest(req);
    process.stdout.write(JSON.stringify(res) + "\n");
  } catch (err) {
    const id = req?.id ?? null;
    process.stdout.write(JSON.stringify({ id, error: String(err?.message || err) }) + "\n");
  }
});

